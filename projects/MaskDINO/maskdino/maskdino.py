# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from mmdet.models import MaskFormer
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class MaskDINO(MaskFormer):
    r"""Implementation of `Mask DINO: Towards A Unified Transformer-based
    Framework for Object Detection and Segmentation
    <https://arxiv.org/abs/2206.02777>`_."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 panoptic_head: OptConfigType = None,
                 panoptic_fusion_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        feats = self.extract_feat(batch_inputs)
        mask_cls_results, mask_pred_results, mask_box_results = self.panoptic_head.predict(
            feats, batch_data_samples)
        results_list = self.panoptic_fusion_head.predict(
            mask_cls_results,
            mask_pred_results,
            mask_box_results,
            batch_data_samples,
            rescale=rescale)
        results = self.add_pred_to_datasample(batch_data_samples, results_list)

        return results
