from torch import Tensor
import torch
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.single_stage import SingleStageDetector
from torch.nn import functional as F
from .utils import retry_if_cuda_oom, sem_seg_postprocess
from mmengine.structures import PixelData
from mmengine.structures import InstanceData
from mmdet.evaluation.functional import INSTANCE_OFFSET


@MODELS.register_module()
class XDecoder(SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 semseg_head: OptConfigType = None,
                 task: str = 'semseg',
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.task = task
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        self.sem_seg_head = None
        if semseg_head is not None:
            semseg_head_ = semseg_head.deepcopy()
            semseg_head_.update(train_cfg=train_cfg)
            semseg_head_.update(test_cfg=test_cfg)
            semseg_head_.update(task=task)
            self.sem_seg_head = MODELS.build(semseg_head_)
        self.test_cfg = test_cfg

        self.return_inter_mask = False
        if self.task == 'ref-captioning':
            self.return_inter_mask = True

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        visual_features = self.extract_feat(batch_inputs)

        if self.sem_seg_head:
            if self.task == 'caption':
                if not hasattr(self, 'start_token'):
                    self.start_token = torch.tensor([[49406] * 77], device=batch_inputs.device)
                outputs = self.sem_seg_head.predict(visual_features, batch_data_samples,
                                                    extra={'start_token': self.start_token})
                for text, data_samples in zip(outputs['pred_caption'], batch_data_samples):
                    data_samples.pred_caption = text

                if 'pred_sem_seg' in batch_data_samples[0]:
                    for img_metas, data_samples in zip(batch_img_metas, batch_data_samples):
                        original_caption = data_samples.text.split('.')
                        text_prompts = list(
                            filter(lambda x: len(x) > 0, original_caption))

                        height = img_metas["ori_shape"][0]
                        width = img_metas["ori_shape"][1]
                        image_size = img_metas["grounding_img_shape"][:2]
                        sem_seg = retry_if_cuda_oom(sem_seg_postprocess)(
                            data_samples.pred_sem_seg.data.float(), image_size, height, width
                        )

                        if sem_seg.shape[0] == 1:
                            sem_seg = sem_seg.squeeze(0) > self.test_cfg.mask_thr
                            label_names = ['background', text_prompts[0]]
                        else:
                            if not self.test_cfg.keep_bg:
                                sem_seg = sem_seg.max(0)[1]
                                label_names = [text_prompts[id] for id in torch.unique(sem_seg)]
                            else:
                                flag = sem_seg > self.test_cfg.mask_thr
                                sem_seg = sem_seg.max(0)[1] + 1
                                label_names = [text_prompts[id] for id in torch.unique(sem_seg) - 1]
                                sem_seg[flag.sum(0) == 0] = 0
                                label_names.insert(0, 'background')

                        pred_sem_seg = PixelData(data=sem_seg, metainfo={'label_names': label_names})
                        data_samples.pred_sem_seg = pred_sem_seg

                return batch_data_samples

            text_prompts = []
            stuff_text_prompts = []
            for data_samples in batch_data_samples:
                original_caption = data_samples.text.split('.')
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))
                text_prompts.append(original_caption)

                if 'stuff_text' in data_samples:
                    original_caption = data_samples.stuff_text.split('.')
                    original_caption = list(
                        filter(lambda x: len(x) > 0, original_caption))
                    stuff_text_prompts.append(original_caption)
                    text_prompts[-1].extend(original_caption)

            # TODO: Fix
            text_prompts = text_prompts[0]

            outputs = self.sem_seg_head.predict(
                visual_features,
                batch_data_samples,
                text_prompts,
                rescale=rescale)

            if self.task in ['semseg', 'instance', 'panoptic']:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(batch_inputs.shape[-2], batch_inputs.shape[-1]),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True
                )
                for mask_cls_result, mask_pred_result, img_metas, data_samples in zip(mask_cls_results,
                                                                                      mask_pred_results,
                                                                                      batch_img_metas,
                                                                                      batch_data_samples):
                    height = img_metas["ori_shape"][0]
                    width = img_metas["ori_shape"][1]
                    image_size = img_metas["img_shape"][:2]

                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )  # 101，h,w,
                    mask_cls_result = mask_cls_result.to(mask_pred_result)  # 101,10，

                    if self.task == 'semseg':
                        sem_seg = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result, False)
                        if sem_seg.shape[0] == 1:
                            sem_seg = sem_seg.squeeze(0) > self.test_cfg.mask_thr
                            label_names = ['background', text_prompts[0]]
                        else:
                            if not self.test_cfg.keep_bg:
                                sem_seg = sem_seg.max(0)[1]
                                label_names = [text_prompts[id] for id in torch.unique(sem_seg)]
                            else:
                                flag = sem_seg > self.test_cfg.mask_thr
                                sem_seg = sem_seg.max(0)[1] + 1
                                label_names = [text_prompts[id] for id in torch.unique(sem_seg) - 1]
                                sem_seg[flag.sum(0) == 0] = 0
                                label_names.insert(0, 'background')

                        pred_sem_seg = PixelData(data=sem_seg, metainfo={'label_names': label_names})
                        data_samples.pred_sem_seg = pred_sem_seg
                    elif self.task == 'instance':
                        pred_instances = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result,
                                                                                    text_prompts)
                        data_samples.pred_instances = pred_instances
                    elif self.task == 'panoptic':
                        stuff_text_prompts = stuff_text_prompts[0]
                        thing_text_prompts = [x for x in text_prompts if x not in stuff_text_prompts]
                        pred_panoptic_seg = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result,
                                                                                       mask_pred_result,
                                                                                       thing_text_prompts,
                                                                                       stuff_text_prompts)
                        data_samples.pred_panoptic_seg =pred_panoptic_seg

            elif self.task == 'ref-semseg':
                mask_pred_results = outputs["pred_masks"]
                for mask_pred_result, img_metas, data_samples in zip(
                        mask_pred_results,
                        batch_img_metas,
                        batch_data_samples):
                    mask_pred_result = F.interpolate(
                        mask_pred_result[None,],
                        size=(batch_inputs.shape[-2], batch_inputs.shape[-1]),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True
                    )[0]

                    if self.return_inter_mask:
                        sem_seg = mask_pred_result > 0
                        pred_sem_seg = PixelData(data=sem_seg)
                        data_samples.pred_sem_seg = pred_sem_seg
                        continue

                    height = img_metas["ori_shape"][0]
                    width = img_metas["ori_shape"][1]
                    image_size = img_metas["img_shape"][:2]
                    sem_seg = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    if sem_seg.shape[0] == 1:
                        sem_seg = sem_seg.squeeze(0) > self.test_cfg.mask_thr
                        label_names = ['background', text_prompts[0]]
                    else:
                        if not self.test_cfg.keep_bg:
                            sem_seg = sem_seg.max(0)[1]
                            label_names = [text_prompts[id] for id in torch.unique(sem_seg)]
                        else:
                            flag = sem_seg > self.test_cfg.mask_thr
                            sem_seg = sem_seg.max(0)[1] + 1
                            label_names = [text_prompts[id] for id in torch.unique(sem_seg) - 1]
                            sem_seg[flag.sum(0) == 0] = 0
                            label_names.insert(0, 'background')

                    pred_sem_seg = PixelData(data=sem_seg, metainfo={'label_names': label_names})
                    data_samples.pred_sem_seg = pred_sem_seg
            elif self.task == 'retrieval':
                batch_data_samples[0].pred_score = outputs['pred_logits']
        return batch_data_samples

    def instance_inference(self, mask_cls, mask_pred, text_prompts):
        # [Q, K]
        num_class = len(text_prompts)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(num_class, device=scores.device).unsqueeze(0).repeat(scores.shape[0], 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_cfg.max_per_img, sorted=False)

        labels_per_image = labels[topk_indices]
        topk_indices = (topk_indices // num_class)
        mask_pred = mask_pred[topk_indices]

        result = InstanceData()
        # mask (before sigmoid)
        result.masks = (mask_pred > 0).float()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.masks.flatten(1)).sum(1) / (
                result.masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.labels = labels_per_image
        result.label_names = [text_prompts[label] for label in labels_per_image]
        return result

    def semantic_inference(self, mask_cls, mask_pred, keep_sem_bgd=False):
        if keep_sem_bgd:
            mask_cls = F.softmax(mask_cls, dim=-1)
        else:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred, thing_text, stuff_text):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(len(thing_text) + len(stuff_text)) & (scores > self.test_cfg.mask_thr)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        label_names = []

        current_segment_id = 1

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = int(pred_class) < len(thing_text)
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.test_cfg.overlap_thr:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        else:
                            stuff_memory_list[int(pred_class)] = int(pred_class)
                            panoptic_seg[mask] = int(pred_class) + 1  # 0 is background
                            label_names.append({int(pred_class)+1: stuff_text[int(pred_class) - len(thing_text)]})
                        continue

                    segment_id = int(pred_class) + 1 + current_segment_id * INSTANCE_OFFSET # 0 is background
                    current_segment_id += 1
                    panoptic_seg[mask] = segment_id
                    label_names.append({segment_id: thing_text[int(pred_class)]})

            label_names.insert(0, {0: 'background'})
            panoptic_seg = PixelData(sem_seg=panoptic_seg.int(), metainfo={'label_names': label_names})
            return panoptic_seg
