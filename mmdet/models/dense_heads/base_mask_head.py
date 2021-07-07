from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseMaskHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self):
        super(BaseMaskHead, self).__init__()

    @abstractmethod
    def loss(self, **kwargs):
        pass

    @abstractmethod
    def get_seg(self, **kwargs):
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_masks (Tensor): Ground truth masks of the image.
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation.
         Args:
             feats (tuple[torch.Tensor]): Multi-level features from the
                 upstream network, each is a 4D-tensor.
             img_metas (list[dict]): List of image information.
             rescale (bool, optional): Whether to rescale the results.
                 Defaults to False.
         Returns:
             # TODO add
         """
        outs = self(feats)

        mask_inputs = outs + (img_metas, rescale)
        segm_results = self.bbox_head.get_masks(*mask_inputs)
        return segm_results
