# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule


class BaseMaskHead(BaseModule, metaclass=ABCMeta):
    """Base class for mask heads used in One-Stage Instance Segmentation."""

    def __init__(self, init_cfg):
        super(BaseMaskHead, self).__init__(init_cfg)

    @abstractmethod
    def loss(self, **kwargs):
        pass

    @abstractmethod
    def get_results(self, **kwargs):
        """Get precessed :obj:`InstanceData` of multiple images."""
        pass

    def forward_train(self,
                      x,
                      gt_labels,
                      gt_masks,
                      img_metas,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      positive_infos=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor] | tuple[Tensor]): Features from FPN.
                Each has a shape (B, C, H, W).
            gt_labels (list[Tensor]): Ground truth labels of all images.
                each has a shape (num_gts,).
            gt_masks (list[Tensor]) : Masks for each bbox, has a shape
                (num_gts, h , w).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                each item has a shape (num_gts, 4).
            gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes to be
                ignored, each item has a shape (num_ignored_gts, 4).
            positive_infos (list[:obj:`InstanceData`], optional): Information
                of positive samples. Used when the label assignment is
                done outside the MaskHead, e.g., in BboxHead in
                YOLACT or CondInst, etc. When the label assignment is done in
                MaskHead, it would be None, like SOLO. All values
                in it should have shape (num_positive_samples, *).

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if positive_infos is None:
            outs = self(x)
        else:
            outs = self(x, positive_infos)

        assert isinstance(outs, tuple), 'Forward results should be a tuple, ' \
                                        'even if only one item is returned'
        loss = self.loss(
            *outs,
            gt_labels=gt_labels,
            gt_masks=gt_masks,
            img_metas=img_metas,
            gt_bboxes=gt_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            positive_infos=positive_infos,
            **kwargs)
        return loss

    def simple_test(self,
                    feats,
                    img_metas,
                    rescale=False,
                    instances_list=None,
                    **kwargs):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            instances_list (list[obj:`InstanceData`], optional): Detection
                results of each image after the post process. Only exist
                if there is a `bbox_head`, like `YOLACT`, `CondInst`, etc.

        Returns:
            list[obj:`InstanceData`]: Instance segmentation \
                results of each image after the post process. \
                Each item usually contains following keys. \

                - scores (Tensor): Classification scores, has a shape
                  (num_instance,)
                - labels (Tensor): Has a shape (num_instances,).
                - masks (Tensor): Processed mask results, has a
                  shape (num_instances, h, w).
        """
        if instances_list is None:
            outs = self(feats)
        else:
            outs = self(feats, instances_list=instances_list)
        mask_inputs = outs + (img_metas, )
        results_list = self.get_results(
            *mask_inputs,
            rescale=rescale,
            instances_list=instances_list,
            **kwargs)
        return results_list

    def onnx_export(self, img, img_metas):
        raise NotImplementedError(f'{self.__class__.__name__} does '
                                  f'not support ONNX EXPORT')
