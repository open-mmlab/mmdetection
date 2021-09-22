# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_head
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class YOLACT(SingleStageDetector):
    """Implementation of `YOLACT <https://arxiv.org/abs/1904.02689>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 segm_head,
                 mask_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(YOLACT, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained, init_cfg)
        self.segm_head = build_head(segm_head)
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        raise NotImplementedError

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # convert Bitmap mask or Polygon Mask to Tensor here
        gt_masks = [
            gt_mask.to_tensor(dtype=torch.uint8, device=img.device)
            for gt_mask in gt_masks
        ]

        x = self.extract_feat(img)

        cls_score, bbox_pred, coeff_pred = self.bbox_head(x)
        bbox_head_loss_inputs = (cls_score, bbox_pred) + (gt_bboxes, gt_labels,
                                                          img_metas)
        losses, sampling_results = self.bbox_head.loss(
            *bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        segm_head_outs = self.segm_head(x[0])
        loss_segm = self.segm_head.loss(segm_head_outs, gt_masks, gt_labels)
        losses.update(loss_segm)

        mask_pred = self.mask_head(x[0], coeff_pred, gt_bboxes, img_metas,
                                   sampling_results)
        loss_mask = self.mask_head.loss(mask_pred, gt_masks, gt_bboxes,
                                        img_metas, sampling_results)
        losses.update(loss_mask)

        # check NaN and Inf
        for loss_name in losses.keys():
            assert torch.isfinite(torch.stack(losses[loss_name]))\
                .all().item(), '{} becomes infinite or NaN!'\
                .format(loss_name)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation."""
        feat = self.extract_feat(img)
        det_bboxes, det_labels, det_coeffs = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bbox, det_label, self.bbox_head.num_classes)
            for det_bbox, det_label in zip(det_bboxes, det_labels)
        ]

        segm_results = self.mask_head.simple_test(
            feat,
            det_bboxes,
            det_labels,
            det_coeffs,
            img_metas,
            rescale=rescale)

        return list(zip(bbox_results, segm_results))

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations."""
        raise NotImplementedError(
            'YOLACT does not support test-time augmentation')
