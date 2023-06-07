import copy
from typing import Sequence

import torch
from mmengine.structures import InstanceData, PixelData
from torch import nn
from torch.nn import functional as F

from mmdet.evaluation.functional import INSTANCE_OFFSET
from mmdet.registry import MODELS
from .utils import (is_lower_torch_version, retry_if_cuda_oom,
                    sem_seg_postprocess)


@MODELS.register_module()
class XDecoderUnifiedhead(nn.Module):

    def __init__(self,
                 in_channels: int,
                 pixel_decoder: nn.Module,
                 transformer_decoder: nn.Module,
                 task: str = 'semseg',
                 test_cfg=None):
        super().__init__()
        self.task = task
        self.test_cfg = test_cfg

        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(in_channels=in_channels)
        self.pixel_decoder = MODELS.build(pixel_decoder_)

        transformer_decoder_ = copy.deepcopy(transformer_decoder)
        transformer_decoder_.update(task=task)
        self.predictor = MODELS.build(transformer_decoder_)

        self.return_inter_mask = False
        if self.task == 'ref-caption':
            # ref-caption = ref-semseg + caption,
            # so we need to return the intermediate mask
            self.return_inter_mask = True

        self._all_text_prompts = None
        self._extra = None
        # TODO: Very trick, for retrieval task
        self._force_not_use_cache = False

    def pre_process(self, batch_data_samples, device):
        extra = {}
        if self.task != 'caption':
            # have text
            all_text_prompts = []
            stuff_text_prompts = []
            for data_samples in batch_data_samples:
                if isinstance(data_samples.text, str):
                    text = data_samples.text.split('.')
                elif isinstance(data_samples.text, Sequence):
                    text = data_samples.text
                else:
                    raise TypeError(
                        'Type pf data_sample.text must be sequence or str')
                text = list(filter(lambda x: len(x) > 0, text))
                all_text_prompts.append(text)

                # for panoptic
                if 'stuff_text' in data_samples:
                    if isinstance(data_samples.stuff_text, str):
                        text = data_samples.stuff_text.split('.')
                    elif isinstance(data_samples.stuff_text, Sequence):
                        text = data_samples.stuff_text
                    else:
                        raise TypeError('Type pf data_sample.stuff_text '
                                        'must be sequence or str')
                    text = list(filter(lambda x: len(x) > 0, text))
                    stuff_text_prompts.append(text)
                    all_text_prompts[-1].extend(text)

            # TODO: support batch
            all_text_prompts = all_text_prompts[0]

            if all_text_prompts != self._all_text_prompts \
                    or self._force_not_use_cache:
                # avoid redundant computation
                self._all_text_prompts = all_text_prompts
                if self.task in ['semseg', 'instance', 'panoptic']:
                    self.predictor.lang_encoder.get_mean_embeds(
                        all_text_prompts + ['background'])
                elif self.task == 'ref-semseg':
                    token_info = self.predictor.lang_encoder.get_text_embeds(
                        all_text_prompts, norm=False)
                    token_emb = token_info['token_emb']
                    tokens = token_info['tokens']
                    query_emb = token_emb[tokens['attention_mask'].bool()]
                    extra['grounding_tokens'] = query_emb[:, None]
                    extra['class_emb'] = token_info['class_emb']
                elif self.task == 'retrieval':
                    token_info = self.predictor.lang_encoder.get_text_embeds(
                        all_text_prompts, norm=True)
                    extra['class_emb'] = token_info['class_emb']
                self._extra = extra
                return extra, all_text_prompts, stuff_text_prompts
            else:
                return self._extra, all_text_prompts, stuff_text_prompts
        else:
            if not hasattr(self, 'start_token'):
                self.start_token = self.predictor.lang_encoder. \
                    get_sot_token(device=device)
            extra['start_token'] = self.start_token
            return extra, None, None

    def predict(self, features, batch_data_samples):
        # multi scale feature
        mask_features, multi_scale_features = self.pixel_decoder(features)

        # pre process
        extra, all_text_prompts, stuff_text_prompts = self.pre_process(
            batch_data_samples, mask_features.device)

        # transformer decoder forward
        predictions = self.predictor(
            multi_scale_features, mask_features, extra=extra)

        # post process
        return self.post_process(predictions, batch_data_samples,
                                 all_text_prompts, stuff_text_prompts)

    def post_process(self, predictions, batch_data_samples, all_text_prompts,
                     stuff_text_prompts):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_input_shape = batch_data_samples[0].metainfo['batch_input_shape']

        if self.task == 'caption':
            for text, data_samples in zip(predictions['pred_caption'],
                                          batch_data_samples):
                data_samples.pred_caption = text

            if 'pred_sem_seg' in batch_data_samples[0]:
                for img_metas, data_samples in zip(batch_img_metas,
                                                   batch_data_samples):
                    original_caption = data_samples.text.split('.')
                    text_prompts = list(
                        filter(lambda x: len(x) > 0, original_caption))

                    height = img_metas['ori_shape'][0]
                    width = img_metas['ori_shape'][1]
                    image_size = img_metas['grounding_img_shape'][:2]

                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        data_samples.pred_sem_seg.sem_seg.float(), image_size,
                        height, width)
                    pred_sem_seg = retry_if_cuda_oom(self._semantic_inference)(
                        None, mask_pred_result, text_prompts)
                    data_samples.pred_sem_seg = pred_sem_seg
        elif self.task in ['semseg', 'instance', 'panoptic']:
            mask_pred_results = predictions['pred_masks']
            mask_cls_results = predictions['pred_logits']
            if is_lower_torch_version():
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(batch_input_shape[-2], batch_input_shape[-1]),
                    mode='bilinear',
                    align_corners=False)
            else:
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(batch_input_shape[-2], batch_input_shape[-1]),
                    mode='bicubic',
                    align_corners=False,
                    antialias=True)

            # used for ref-caption
            if self.return_inter_mask:
                sem_seg = mask_pred_results[0] > 0
                pred_sem_seg = PixelData(sem_seg=sem_seg)
                batch_data_samples[0].pred_sem_seg = pred_sem_seg
                return batch_data_samples

            # for batch
            for mask_cls_result, \
                    mask_pred_result, \
                    img_metas, \
                    data_samples in zip(
                                    mask_cls_results,
                                    mask_pred_results,
                                    batch_img_metas,
                                    batch_data_samples):
                height = img_metas['ori_shape'][0]
                width = img_metas['ori_shape'][1]
                image_size = img_metas['img_shape'][:2]
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width)
                mask_cls_result = mask_cls_result.to(mask_pred_result)

                if self.task == 'semseg':
                    pred_sem_seg = retry_if_cuda_oom(self._semantic_inference)(
                        mask_cls_result, mask_pred_result, all_text_prompts)
                    data_samples.pred_sem_seg = pred_sem_seg
                elif self.task == 'instance':
                    pred_instances = retry_if_cuda_oom(
                        self._instance_inference)(mask_cls_result,
                                                  mask_pred_result,
                                                  all_text_prompts)
                    data_samples.pred_instances = pred_instances
                elif self.task == 'panoptic':
                    # TODO: support batch
                    stuff_text_prompts = stuff_text_prompts[0]
                    thing_text_prompts = [
                        x for x in all_text_prompts
                        if x not in stuff_text_prompts
                    ]
                    pred_panoptic_seg = retry_if_cuda_oom(
                        self._panoptic_inference)(mask_cls_result,
                                                  mask_pred_result,
                                                  thing_text_prompts,
                                                  stuff_text_prompts)
                    data_samples.pred_panoptic_seg = pred_panoptic_seg
        elif self.task == 'ref-semseg':
            mask_pred_results = predictions['pred_masks']
            for mask_pred_result, img_metas, data_samples in zip(
                    mask_pred_results, batch_img_metas, batch_data_samples):
                if is_lower_torch_version():
                    mask_pred_result = F.interpolate(
                        mask_pred_result[None, ],
                        size=(batch_input_shape[-2], batch_input_shape[-1]),
                        mode='bilinear',
                        align_corners=False)[0]
                else:
                    mask_pred_result = F.interpolate(
                        mask_pred_result[None, ],
                        size=(batch_input_shape[-2], batch_input_shape[-1]),
                        mode='bicubic',
                        align_corners=False,
                        antialias=True)[0]

                if self.return_inter_mask:
                    sem_seg = mask_pred_result > 0
                    pred_sem_seg = PixelData(sem_seg=sem_seg)
                    data_samples.pred_sem_seg = pred_sem_seg
                    continue

                height = img_metas['ori_shape'][0]
                width = img_metas['ori_shape'][1]
                image_size = img_metas['img_shape'][:2]
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width)

                pred_sem_seg = retry_if_cuda_oom(self._semantic_inference)(
                    None, mask_pred_result, all_text_prompts)
                data_samples.pred_sem_seg = pred_sem_seg
        elif self.task == 'retrieval':
            batch_data_samples[0].pred_score = predictions['pred_logits']
        return batch_data_samples

    def _instance_inference(self, mask_cls, mask_pred, text_prompts):
        num_class = len(text_prompts)

        scores = F.softmax(mask_cls, dim=-1)[:, :-1]

        labels = torch.arange(
            num_class,
            device=scores.device).unsqueeze(0).repeat(scores.shape[0],
                                                      1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            self.test_cfg.max_per_img, sorted=False)

        labels_per_image = labels[topk_indices]
        topk_indices = (topk_indices // num_class)
        mask_pred = mask_pred[topk_indices]

        result = InstanceData()
        result.masks = (mask_pred > 0).float()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) *
                                 result.masks.flatten(1)).sum(1) / (
                                     result.masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.labels = labels_per_image
        result.label_names = [
            text_prompts[label] for label in labels_per_image
        ]
        result.bboxes = result.scores.new_zeros(len(result.scores), 4)
        return result

    def _semantic_inference(self, mask_cls, mask_pred, text_prompts):
        if mask_cls is None:
            sem_seg = mask_pred
        else:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            sem_seg = torch.einsum('qc,qhw->chw', mask_cls, mask_pred)

        if sem_seg.shape[0] == 1:
            sem_seg = sem_seg.squeeze(0) > self.test_cfg.mask_thr
            # the value 0 means background, 1 means foreground
            label_names = text_prompts  # for visualization
        else:
            # 0 ~ num_class, the value 0 means background
            if self.test_cfg.use_thr_for_mc:
                foreground_flag = sem_seg > self.test_cfg.mask_thr
                sem_seg = sem_seg.max(0)[1] + 1
                label_names = [
                    text_prompts[id] for id in torch.unique(sem_seg) - 1
                ]
                sem_seg[foreground_flag.sum(0) == 0] = 0
            else:
                sem_seg = sem_seg.max(0)[1] + 1
                label_names = [
                    text_prompts[id] for id in torch.unique(sem_seg) - 1
                ]
        pred_sem_seg = PixelData(
            sem_seg=sem_seg, metainfo={'label_names': label_names})
        return pred_sem_seg

    def _panoptic_inference(self, mask_cls, mask_pred, thing_text, stuff_text):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(len(thing_text) + len(stuff_text)) & (
            scores > self.test_cfg.mask_thr)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w),
                                   dtype=torch.int32,
                                   device=cur_masks.device)
        label_names = []

        current_segment_id = 1

        if cur_masks.shape[0] == 0:
            pass
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = int(pred_class) < len(thing_text)
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item(
                ) > 0:
                    if mask_area / original_area < self.test_cfg.overlap_thr:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(
                                pred_class)]
                        else:
                            stuff_memory_list[int(pred_class)] = int(
                                pred_class)
                            panoptic_seg[mask] = int(
                                pred_class) + 1  # 0 is background
                            label_names.append({
                                int(pred_class) + 1:
                                stuff_text[int(pred_class) - len(thing_text)]
                            })
                        continue

                    # 0 is background
                    segment_id = int(
                        pred_class) + 1 + current_segment_id * INSTANCE_OFFSET
                    current_segment_id += 1
                    panoptic_seg[mask] = segment_id
                    label_names.append(
                        {segment_id: thing_text[int(pred_class)]})

            label_names.insert(0, {0: 'background'})
            panoptic_seg = PixelData(
                sem_seg=panoptic_seg.int(),
                metainfo={'label_names': label_names})
            return panoptic_seg
