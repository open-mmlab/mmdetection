# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule

from ...builder import build_loss


class BasePanopticFusionHead(BaseModule, metaclass=ABCMeta):
    """Base class for panoptic heads."""

    def __init__(self,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 test_cfg=None,
                 loss_panoptic=None,
                 init_cfg=None,
                 **kwargs):
        super(BasePanopticFusionHead, self).__init__(init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_things_classes + num_stuff_classes
        self.test_cfg = test_cfg

        if loss_panoptic:
            self.loss_panoptic = build_loss(loss_panoptic)
        else:
            self.loss_panoptic = None

    @property
    def with_loss(self):
        """bool: whether the panoptic head contains loss function."""
        return self.loss_panoptic is not None

    @abstractmethod
    def forward_train(self, gt_masks=None, gt_semantic_seg=None, **kwargs):
        """Forward function during training."""

    @abstractmethod
    def simple_test(self,
                    img_metas,
                    det_labels,
                    mask_preds,
                    seg_preds,
                    det_bboxes,
                    cfg=None,
                    **kwargs):
        """Test without augmentation."""
