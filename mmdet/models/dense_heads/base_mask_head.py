from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule


class BaseMaskHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg):
        super(BaseMaskHead, self).__init__(init_cfg)

    @abstractmethod
    def loss(self, **kwargs):
        pass

    @abstractmethod
    def get_masks(self, **kwargs):
        pass

    def forward_train(self,
                      x,
                      gt_labels,
                      gt_masks,
                      img_metas,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      **kwargs):

        outs = self(x)
        loss = self.loss(
            *outs,
            gt_labels,
            gt_masks,
            img_metas,
            gt_bboxes=gt_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            **kwargs)
        return loss

    def simple_test(self, feats, img_metas, rescale=False, **kwargs):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[obj:`InstanceResults`]: Each `InstanceResults` containing
            the following keys.
                - `masks`: has shape (num_masks, H, W)
                - `scores`: has shape (num_masks,)
                - `labels`: has shape (num_masks)
        """
        outs = self(feats)
        mask_inputs = outs + (img_metas, )
        mask_results_list = self.get_masks(
            *mask_inputs, rescale=rescale, **kwargs)
        return mask_results_list
