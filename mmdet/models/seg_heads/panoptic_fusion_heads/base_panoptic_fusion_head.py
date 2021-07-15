from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule

from ...builder import build_loss


class BasePanopticFusionHead(BaseModule, metaclass=ABCMeta):
    """Base class for panoptic heads."""

    def __init__(self,
                 num_things=80,
                 num_stuff=53,
                 loss_panoptic=None,
                 init_cfg=None,
                 **kwargs):
        super(BasePanopticFusionHead, self).__init__(init_cfg)
        self.num_things = num_things
        self.num_stuff = num_stuff

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
                    seg_logits,
                    det_bboxes,
                    cfg=None,
                    **kwargs):
        """Test without augmentation."""
