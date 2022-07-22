# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .faster_rcnn import FasterRCNN


@MODELS.register_module()
class TridentFasterRCNN(FasterRCNN):
    """Implementation of `TridentNet <https://arxiv.org/abs/1901.01892>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        assert self.backbone.num_branch == self.roi_head.num_branch
        assert self.backbone.test_branch_idx == self.roi_head.test_branch_idx
        self.num_branch = self.backbone.num_branch
        self.test_branch_idx = self.backbone.test_branch_idx

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """copy the ``batch_data_samples`` to fit multi-branch."""
        num_branch = self.num_branch \
            if self.training or self.test_branch_idx == -1 else 1
        trident_data_samples = batch_data_samples * num_branch
        return super()._forward(
            batch_inputs=batch_inputs, batch_data_samples=trident_data_samples)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """copy the ``batch_data_samples`` to fit multi-branch."""
        num_branch = self.num_branch \
            if self.training or self.test_branch_idx == -1 else 1
        trident_data_samples = batch_data_samples * num_branch
        return super().loss(
            batch_inputs=batch_inputs, batch_data_samples=trident_data_samples)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """copy the ``batch_data_samples`` to fit multi-branch."""
        num_branch = self.num_branch \
            if self.training or self.test_branch_idx == -1 else 1
        trident_data_samples = batch_data_samples * num_branch
        return super().predict(
            batch_inputs=batch_inputs,
            batch_data_samples=trident_data_samples,
            rescale=rescale)

    # TODO need to refactor
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        num_branch = (self.num_branch if self.test_branch_idx == -1 else 1)
        trident_img_metas = [img_metas * num_branch for img_metas in img_metas]
        proposal_list = self.rpn_head.aug_test_rpn(x, trident_img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
