# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F

from mmdet.core import bbox2result, multi_apply
from mmdet.models.builder import DETECTORS, build_head
from mmdet.models.utils import preprocess_panoptic_gt
from .panoptic_two_stage_segmentor import (TwoStageDetector,
                                           TwoStagePanopticSegmentor)


@DETECTORS.register_module()
class KNet(TwoStagePanopticSegmentor):
    r"""Implementation of `K-Net: Towards Unified Image Segmentation
    <https://arxiv.org/pdf/2106.14855>`_."""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 panoptic_fusion_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        assert self.with_rpn, 'KNet does not support external proposals'
        # check strides in forward_train
        self.mask_assign_out_stride = self.roi_head.mask_assign_out_stride
        self.feat_scale_factor = self.rpn_head.feat_scale_factor
        self.output_level = self.rpn_head.localization_fpn_cfg.output_level

        self.num_things_classes = self.roi_head.num_things_classes
        self.num_stuff_classes = self.roi_head.num_stuff_classes
        self.num_classes = self.roi_head.num_classes

        panoptic_cfg = test_cfg.fusion
        panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
        panoptic_fusion_head_.update(test_cfg=panoptic_cfg)
        self.panoptic_fusion_head = build_head(panoptic_fusion_head_)

    def preprocess_gt(self, gt_labels_list, gt_masks_list, gt_semantic_segs,
                      img_metas):
        """Preprocess the ground truth for all images.

        Args:
            gt_labels_list (list[Tensor]): Each is ground truth
                labels of each bbox, with shape (num_gts, ).
            gt_masks_list (list[BitmapMasks]): Each is ground truth
                masks of each instances of a image, shape
                (num_gts, h, w).
            gt_semantic_seg (Tensor | None): Ground truth of semantic
                segmentation with the shape (batch_size, n, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.

            - gt_labels_list (list[Tensor]): Ground truth class indices
              for all images. Each with shape (n, ), n is the number
              of instance in a image.
            - gt_masks_list (list[Tensor]): Ground truth mask of instances
              for each image, each with shape (n, h, w).
            - gt_sem_labels_list (list[Tensor]): Ground truth class indices
              for all images. Each with shape (n, ), n is the number
              of stuff class in a image.
            - gt_sem_masks_list (list[Tensor]): Ground truth mask of stuff
              for all images. Each with shape (n, h, w).
        """
        num_things_list = [self.num_things_classes] * len(gt_labels_list)
        num_stuff_list = [self.num_stuff_classes] * len(gt_labels_list)

        if gt_semantic_segs is None:
            gt_semantic_segs = [None] * len(gt_labels_list)

        targets = multi_apply(
            preprocess_panoptic_gt,
            gt_labels_list,
            gt_masks_list,
            gt_semantic_segs,
            num_things_list,
            num_stuff_list,
            img_metas,
            merge_things_stuff=False)
        (gt_labels_list, gt_masks_list, gt_sem_labels_list,
         gt_sem_masks_list) = targets

        pad_H, pad_W = img_metas[0]['batch_input_shape']
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

        return (gt_labels_list, gt_masks_list, gt_sem_labels_list,
                gt_sem_masks_list)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (batch_size, c, h, w) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[Dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            gt_masks (list[BitmapMasks]): true segmentation masks for each box
                used if the architecture supports a segmentation task.
            gt_semantic_seg (list[tensor] or None): semantic segmentation mask
                for images for panoptic segmentation.
                Defaults to None for instance segmentation.
            gt_bboxes_ignore (list[Tensor] or None): specify which bounding
                boxes can be ignored when computing the loss.
                Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # add batch_input_shape in img_metas
        super(TwoStageDetector, self).forward_train(img, img_metas)

        gt_labels, gt_masks, gt_sem_cls, gt_sem_seg = self.preprocess_gt(
            gt_labels, gt_masks, gt_semantic_seg, img_metas)
        x = self.extract_feat(img)

        output_level_stride = img_metas[0]['batch_input_shape'][-1] // x[
            self.output_level].shape[-1]
        assert (output_level_stride == self.mask_assign_out_stride *
                self.feat_scale_factor), 'Stride of output_level' \
            'should be equal to mask_assign_out_stride * ' \
            'feat_scale_factor'

        rpn_results = self.rpn_head.forward_train(
            x=x,
            img_metas=img_metas,
            gt_masks=gt_masks,
            gt_labels=gt_labels,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls)
        (rpn_losses, x_feats, proposal_feats, mask_preds) = rpn_results

        losses = self.roi_head.forward_train(
            x=x_feats,
            proposal_feats=proposal_feats,
            mask_preds=mask_preds,
            img_metas=img_metas,
            gt_masks=gt_masks,
            gt_labels=gt_labels,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls)

        losses.update(rpn_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            imgs (Tensor): A batch of images.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[dict[str, tuple[list]] | tuple[list]]:
                Panoptic segmentation results of each image for panoptic
                segmentation, or formatted bbox and mask results of each
                image for instance segmentation.

            .. code-block:: none

                [
                    # panoptic segmentation
                    {
                        'pan_results': np.array, # shape = [h, w]
                        'ins_results': tuple[list],
                        # semantic segmentation results are not supported yet
                        'sem_results': np.array
                    },
                    ...
                ]

            or

            .. code-block:: none

                [
                    # instance segmentation
                    (
                        bboxes, # list[np.array]
                        masks # list[list[np.array]]
                    ),
                    ...
                ]
        """
        x = self.extract_feat(img)

        rpn_results = self.rpn_head.simple_test_rpn(x=x, img_metas=img_metas)
        (x_feats, proposal_feats, mask_preds) = rpn_results

        mask_cls, mask_preds = self.roi_head.simple_test(
            x=x_feats, proposal_feats=proposal_feats, mask_preds=mask_preds)

        results = self.panoptic_fusion_head.simple_test(
            mask_cls_results=mask_cls,
            mask_pred_results=mask_preds,
            img_metas=img_metas,
            rescale=rescale)

        for i in range(len(results)):
            if 'pan_results' in results[i]:
                results[i]['pan_results'] = results[i]['pan_results'].detach(
                ).cpu().numpy()

            if 'ins_results' in results[i]:
                labels_per_image, bboxes, mask_pred_binary = results[i][
                    'ins_results']
                bbox_results = bbox2result(bboxes, labels_per_image,
                                           self.num_things_classes)
                mask_results = [[] for _ in range(self.num_things_classes)]
                for j, label in enumerate(labels_per_image):
                    mask = mask_pred_binary[j].detach().cpu().numpy()
                    mask_results[label].append(mask)
                results[i]['ins_results'] = bbox_results, mask_results

        if self.num_stuff_classes == 0:
            results = [res['ins_results'] for res in results]

        return results
