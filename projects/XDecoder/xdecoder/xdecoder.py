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

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        visual_features = self.extract_feat(batch_inputs)

        if self.sem_seg_head:
            if self.task == 'captioning':
                if not hasattr(self, 'start_token'):
                    self.start_token = torch.tensor([[49406] * 77], device=batch_inputs.device)
                outputs = self.sem_seg_head.predict(visual_features, batch_data_samples, extra={'start_token': self.start_token})
                print(outputs)
                return outputs

            text_prompts = []
            for data_samples in batch_data_samples:
                original_caption = data_samples.caption.split('.')
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))
                text_prompts.append(original_caption)

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
                            sem_seg = sem_seg.squeeze(0) > 0.5
                        else:
                            # flag = sem_seg > 0.5
                            # sem_seg = sem_seg.max(0)[1] + 1
                            # sem_seg[flag.sum(0) == 0] = 0
                            sem_seg = sem_seg.max(0)[1]
                        # pred_sem_seg = PixelData(metainfo={'bg_value': 0}, data=sem_seg)
                        pred_sem_seg = PixelData(data=sem_seg)
                        data_samples.pred_sem_seg = pred_sem_seg
                    elif self.task == 'instance':
                        pred_instances = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result,
                                                                                    len(text_prompts))
                        data_samples.pred_instances = pred_instances
                    elif self.task == 'panoptic':

                        data_samples.pred_panoptic_seg = pred_panoptic_seg

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
                    height = img_metas["ori_shape"][0]
                    width = img_metas["ori_shape"][1]
                    image_size = img_metas["img_shape"][:2]
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    if mask_pred_result.shape[0] == 1:
                        sem_seg = mask_pred_result > 0
                    else:
                        # flag = mask_pred_result > 0
                        # sem_seg = mask_pred_result.max(0)[1]+1
                        sem_seg = mask_pred_result.max(0)[1]
                        # sem_seg[flag.sum(0) == 0] = 0
                    # pred_sem_seg = PixelData(metainfo={'bg_value': 0}, data=sem_seg)
                    pred_sem_seg = PixelData(data=sem_seg)
                    data_samples.pred_sem_seg = pred_sem_seg
        return batch_data_samples

    def instance_inference(self, mask_cls, mask_pred, num_class):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(num_class, device=scores.device).unsqueeze(0).repeat(scores.shape[0], 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_cfg.max_per_img, sorted=False)

        labels_per_image = labels[topk_indices]
        topk_indices = (topk_indices // num_class)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        # if self.panoptic_on:
        #     thing_dataset_id_to_contiguous_id = self.metadata.thing_dataset_id_to_contiguous_id if hasattr(
        #         self.metadata, 'thing_dataset_id_to_contiguous_id') else {}
        #     keep = torch.zeros_like(scores_per_image).bool()
        #     for i, lab in enumerate(labels_per_image):
        #         keep[i] = lab in thing_dataset_id_to_contiguous_id.values()
        #
        #     scores_per_image = scores_per_image[keep]
        #     labels_per_image = labels_per_image[keep]
        #     mask_pred = mask_pred[keep]

        result = InstanceData()
        # mask (before sigmoid)
        result.masks = (mask_pred > 0).float()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.masks.flatten(1)).sum(1) / (
                result.masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.labels = labels_per_image

        return result

    def semantic_inference(self, mask_cls, mask_pred, keep_sem_bgd=False):
        if keep_sem_bgd:
            mask_cls = F.softmax(mask_cls, dim=-1)
        else:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            thing_dataset_id_to_contiguous_id = self.metadata.thing_dataset_id_to_contiguous_id if hasattr(
                self.metadata, 'thing_dataset_id_to_contiguous_id') else {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
            return panoptic_seg, segments_info
