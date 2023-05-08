# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from ..utils import multi_apply, preprocess_panoptic_gt
from .panoptic_two_stage_segmentor import TwoStagePanopticSegmentor


@MODELS.register_module()
class KNet(TwoStagePanopticSegmentor):
    r"""Implementation of `K-Net: Towards Unified Image Segmentation
    <https://arxiv.org/pdf/2106.14855>`_."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 panoptic_fusion_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        assert self.with_rpn, 'KNet does not support external proposals'

        self.mask_assign_out_stride = self.roi_head.mask_assign_out_stride
        self.feat_scale_factor = self.rpn_head.feat_scale_factor
        self.output_level = self.rpn_head.localization_fpn_cfg.output_level

        self.num_things_classes = self.roi_head.num_things_classes
        self.num_stuff_classes = self.roi_head.num_stuff_classes
        self.num_classes = self.roi_head.num_classes

        panoptic_cfg = test_cfg.fusion
        panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
        panoptic_fusion_head_.update(test_cfg=panoptic_cfg)
        self.panoptic_fusion_head = MODELS.build(panoptic_fusion_head_)

    def preprocess_gt(self, batch_data_samples: SampleList) -> SampleList:
        """Preprocess the ground truth for all images."""
        num_things_list = [self.num_things_classes] * len(batch_data_samples)
        num_stuff_list = [self.num_stuff_classes] * len(batch_data_samples)
        gt_labels_list = []
        gt_masks_list = []
        gt_senmantic_seg_list = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            gt_labels_list.append(data_sample.gt_instances.labels)
            gt_masks_list.append(data_sample.gt_instances.masks)
            batch_img_metas.append(data_sample.metainfo)
            if 'gt_sem_seg' in data_sample:
                gt_senmantic_seg_list.append(data_sample.gt_sem_seg.sem_seg)
            else:
                gt_senmantic_seg_list.append(None)

        targets = multi_apply(
            preprocess_panoptic_gt,
            gt_labels_list,
            gt_masks_list,
            gt_senmantic_seg_list,
            num_things_list,
            num_stuff_list,
            merge_things_stuff=False)
        (gt_labels_list, gt_masks_list, gt_sem_labels_list,
         gt_sem_masks_list) = targets

        pad_H, pad_W = batch_img_metas[0]['batch_input_shape']
        assign_H = pad_H // self.mask_assign_out_stride
        assign_W = pad_W // self.mask_assign_out_stride

        gt_masks_list = [
            F.interpolate(
                gt_masks.unsqueeze(1).float(), (assign_H, assign_W),
                mode='bilinear',
                align_corners=False).squeeze(1) for gt_masks in gt_masks_list
        ]

        if gt_sem_masks_list[0] is not None:
            gt_sem_masks_list = [
                F.interpolate(
                    gt_sem_masks.unsqueeze(1).float(), (assign_H, assign_W),
                    mode='bilinear',
                    align_corners=False).squeeze(1)
                for gt_sem_masks in gt_sem_masks_list
            ]

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.gt_instances = InstanceData(
                labels=gt_labels_list[i], masks=gt_masks_list[i])
            if gt_sem_labels_list[i] is not None:
                data_sample.gt_sem_instances = InstanceData(
                    labels=gt_sem_labels_list[i], masks=gt_sem_masks_list[i])

        return batch_data_samples

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        batch_data_samples = self.preprocess_gt(batch_data_samples)
        x = self.extract_feat(batch_inputs)

        output_level_stride = batch_data_samples[0].batch_input_shape[-1] // x[
            self.output_level].shape[-1]
        assert (output_level_stride == self.mask_assign_out_stride *
                self.feat_scale_factor), 'Stride of output_level' \
            'should be equal to mask_assign_out_stride * ' \
            'feat_scale_factor'

        rpn_results = self.rpn_head.loss(
            x=x, batch_data_samples=batch_data_samples)
        (rpn_losses, x_feats, proposal_feats, mask_preds) = rpn_results

        losses = self.roi_head.loss(
            x=x_feats,
            proposal_feats=proposal_feats,
            mask_preds=mask_preds,
            batch_data_samples=batch_data_samples)

        losses.update(rpn_losses)
        return losses

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: List[dict]) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (List[dict]): Instance segmentation, segmantic
                segmentation and panoptic segmentation results.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """
        for data_sample, pred_results in zip(data_samples, results_list):
            if 'pan_results' in pred_results:
                data_sample.pred_panoptic_seg = pred_results['pan_results']

            if 'ins_results' in pred_results:
                data_sample.pred_instances = pred_results['ins_results']

            assert 'sem_results' not in pred_results, 'segmantic ' \
                'segmentation results are not supported yet.'

        return data_samples

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        x = self.extract_feat(batch_inputs)
        rpn_results = self.rpn_head.predict(
            x=x, batch_data_samples=batch_data_samples)
        x_feats, proposal_feats, mask_preds = rpn_results

        mask_cls, mask_pred = self.roi_head.predict(
            x=x_feats, proposal_feats=proposal_feats, mask_preds=mask_preds)

        results_list = self.panoptic_fusion_head.predict(
            mask_cls_results=mask_cls,
            mask_pred_results=mask_pred,
            batch_data_samples=batch_data_samples,
            rescale=rescale)
        results = self.add_pred_to_datasample(batch_data_samples, results_list)
        return results

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        x = self.extract_feat(batch_inputs)
        rpn_results = self.rpn_head.predict(
            x=x, batch_data_samples=batch_data_samples)
        x_feats, proposal_feats, mask_preds = rpn_results

        mask_cls, mask_pred = self.roi_head.predict(
            x=x_feats, proposal_feats=proposal_feats, mask_preds=mask_preds)
        return mask_cls, mask_pred

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        # TODO remove
        old_keys = [
            key for key in state_dict.keys()
            if 'rpn_head.localization_fpn.aux_convs.0' in key
        ]
        for key in old_keys:
            state_dict[key.replace('s.0', '')] = state_dict.pop(key)
        return super()._load_from_state_dict(state_dict, prefix,
                                             local_metadata, strict,
                                             missing_keys, unexpected_keys,
                                             error_msgs)
