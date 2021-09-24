# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule


class BaseMaskHead(BaseModule, metaclass=ABCMeta):
    """Base class for heads used in One-Stage Instance Segmentation."""

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
                Each has shape (B, C, H, W).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_masks (None | Tensor) : Masks for each bbox, shape
                (num_gts, h , w).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            positive_infos (:obj:`InstanceData`): Only exist
                when there is a `bbox_head` in `SingleStageInstanceSegmentor`
                like `YOLACT`, `CondInst`, etc. It contains the
                information of positive samples.
                If there is only `mask_head` in `SingleStageInstanceSegmentor`,
                it would be None, like SOLO. All values in it should have
                shape (num_positive, *).

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
                    det_results=None,
                    **kwargs):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            det_results (list[obj:`InstanceData`]): Detection
                results of each image after the post process. Only exist
                if there is a `bbox_head`, like `YOLACT`, `CondInst`, etc.

        Returns:
            list[obj:`InstanceData`]: Instance segmentation \
                results of each image after the post process. \
                Each item usually contains following keys. \

                - scores (Tensor): Classification scores, has shape
                  (num_instance,)
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """
        if det_results is None:
            outs = self(feats)
        else:
            outs = self(feats, det_results=det_results)
        mask_inputs = outs + (img_metas, )
        results_list = self.get_results(
            *mask_inputs, rescale=rescale, det_results=det_results, **kwargs)
        return results_list

    def onnx_export(self, img, img_metas):
        raise NotImplementedError(f'{self.__class__.__name__} does '
                                  f'not support ONNX EXPORT')
