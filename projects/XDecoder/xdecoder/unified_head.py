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
            # ref-caption = ref-seg + caption,
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
            num_thing_class = 0
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
                num_thing_class = len(text)
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
                elif self.task == 'ref-seg':
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
                return extra, all_text_prompts, num_thing_class
            else:
                return self._extra, all_text_prompts, num_thing_class
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
        extra, all_text_prompts, num_thing_class = self.pre_process(
            batch_data_samples, mask_features.device)

        # transformer decoder forward
        predictions = self.predictor(
            multi_scale_features, mask_features, extra=extra)

        # post process
        return self.post_process(predictions, batch_data_samples,
                                 all_text_prompts, num_thing_class)

    def post_process(self, predictions, batch_data_samples, all_text_prompts,
                     num_thing_class):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_input_shape = batch_data_samples[0].metainfo['batch_input_shape']

        if self.task == 'caption':
            for text, data_samples in zip(predictions['pred_caption'],
                                          batch_data_samples):
                data_samples.pred_caption = text

            if 'pred_instances' in batch_data_samples[0]:
                for img_metas, data_samples in zip(batch_img_metas,
                                                   batch_data_samples):
                    original_caption = data_samples.text.split('.')
                    text_prompts = list(
                        filter(lambda x: len(x) > 0, original_caption))

                    height = img_metas['ori_shape'][0]
                    width = img_metas['ori_shape'][1]
                    image_size = img_metas['grounding_img_shape'][:2]

                    mask_pred_result = data_samples.pred_instances.masks.float(
                    )
                    mask_cls_result = data_samples.pred_instances.scores.float(
                    )

                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width)

                    pred_instances = retry_if_cuda_oom(
                        self._instance_inference)(mask_cls_result,
                                                  mask_pred_result,
                                                  text_prompts)
                    data_samples.pred_instances = pred_instances

        elif self.task in ['semseg', 'instance', 'panoptic']:
            mask_pred_results = predictions['pred_masks']
            mask_cls_results = predictions['pred_logits']
            if is_lower_torch_version():
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(batch_input_shape[-2], batch_input_shape[-1]),
                    mode='bicubic',
                    align_corners=False)
            else:
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(batch_input_shape[-2], batch_input_shape[-1]),
                    mode='bicubic',
                    align_corners=False,
                    antialias=True)

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
                    pred_panoptic_seg = retry_if_cuda_oom(
                        self._panoptic_inference)(mask_cls_result,
                                                  mask_pred_result,
                                                  all_text_prompts,
                                                  num_thing_class)
                    data_samples.pred_panoptic_seg = pred_panoptic_seg
        elif self.task == 'ref-seg':
            mask_pred_results = predictions['pred_masks']
            mask_cls_results = predictions['pred_logits']
            results_ = zip(mask_pred_results, mask_cls_results,
                           batch_img_metas, batch_data_samples)
            for mask_pred_result, mask_cls_result, \
                    img_metas, data_samples in results_:
                if is_lower_torch_version():
                    mask_pred_result = F.interpolate(
                        mask_pred_result[None],
                        size=(batch_input_shape[-2], batch_input_shape[-1]),
                        mode='bicubic',
                        align_corners=False)[0]
                else:
                    mask_pred_result = F.interpolate(
                        mask_pred_result[None],
                        size=(batch_input_shape[-2], batch_input_shape[-1]),
                        mode='bicubic',
                        align_corners=False,
                        antialias=True)[0]

                if self.return_inter_mask:
                    mask = mask_pred_result > 0
                    pred_instances = InstanceData()
                    pred_instances.masks = mask
                    pred_instances.scores = mask_cls_result
                    data_samples.pred_instances = pred_instances
                    continue

                height = img_metas['ori_shape'][0]
                width = img_metas['ori_shape'][1]
                image_size = img_metas['img_shape'][:2]
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width)

                pred_instances = retry_if_cuda_oom(self._instance_inference)(
                    mask_cls_result, mask_pred_result, all_text_prompts)
                data_samples.pred_instances = pred_instances
        elif self.task == 'retrieval':
            batch_data_samples[0].pred_score = predictions['pred_logits']
        return batch_data_samples

    def _instance_inference(self, mask_cls, mask_pred, text_prompts):
        num_class = len(text_prompts)

        if self.task in ['ref-seg', 'caption']:
            scores = F.softmax(mask_cls, dim=-1)
            scores_per_image = scores.max(dim=-1)[0]
            labels_per_image = torch.arange(num_class)
        else:
            scores = F.softmax(mask_cls, dim=-1)[:, :-1]

            labels = torch.arange(
                num_class,
                device=scores.device).unsqueeze(0).repeat(scores.shape[0],
                                                          1).flatten(0, 1)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(
                self.test_cfg.get('max_per_img', 100), sorted=False)

            labels_per_image = labels[topk_indices]
            topk_indices = (topk_indices // num_class)
            mask_pred = mask_pred[topk_indices]

        result = InstanceData()
        mask_pred = mask_pred.sigmoid()
        result.masks = (mask_pred > self.test_cfg.mask_thr).float()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.flatten(1) *
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
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        sem_seg = torch.einsum('qc,qhw->chw', mask_cls, mask_pred)

        if sem_seg.shape[0] == 1:
            # 0 is foreground, ignore_index is background
            sem_seg = (sem_seg.squeeze(0) <= self.test_cfg.mask_thr).int()
            sem_seg[sem_seg == 1] = self.test_cfg.get('ignore_index', 255)
        else:
            # 0 is foreground, ignore_index is background
            if self.test_cfg.use_thr_for_mc:
                foreground_flag = sem_seg > self.test_cfg.mask_thr
                sem_seg = sem_seg.max(0)[1]
                sem_seg[foreground_flag.sum(0) == 0] = self.test_cfg.get(
                    'ignore_index', 255)
            else:
                sem_seg = sem_seg.max(0)[1]
        pred_sem_seg = PixelData(
            sem_seg=sem_seg[None],
            metainfo={
                'label_names': text_prompts,
                'ignore_index': self.test_cfg.get('ignore_index', 255)
            })
        return pred_sem_seg

    def _panoptic_inference(self, mask_cls, mask_pred, all_text_prompts,
                            num_thing_class):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(len(all_text_prompts)) & (
            scores > self.test_cfg.mask_thr)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.test_cfg.get('ignore_index', 255),
                                  dtype=torch.int32,
                                  device=cur_masks.device)
        instance_id = 1

        if cur_masks.shape[0] > 0:
            cur_mask_ids = cur_prob_masks.argmax(0)
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = int(pred_class) < num_thing_class
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item(
                ) > 0:
                    if mask_area / original_area < self.test_cfg.overlap_thr:
                        continue
                    # merge stuff regions
                    if not isthing:
                        panoptic_seg[mask] = int(pred_class)
                    else:
                        panoptic_seg[mask] = int(
                            pred_class) + instance_id * INSTANCE_OFFSET
                        instance_id += 1

        panoptic_seg = PixelData(
            sem_seg=panoptic_seg[None],
            metainfo={
                'label_names': all_text_prompts,
                'ignore_index': self.test_cfg.get('ignore_index', 255)
            })
        return panoptic_seg
