# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend  # noqa
from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmcv.ops import point_sample, rel_roi_point_to_rel_img_point
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class PointRendRoIHead(StandardRoIHead):
    """`PointRend <https://arxiv.org/abs/1912.08193>`_."""

    def __init__(self, point_head: ConfigType, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.with_bbox and self.with_mask
        self.init_point_head(point_head)

    def init_point_head(self, point_head: ConfigType) -> None:
        """Initialize ``point_head``"""
        self.point_head = MODELS.build(point_head)

    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], bbox_feats: Tensor,
                  batch_gt_instances: InstanceList) -> dict:
        """Run forward function and calculate loss for mask head and point head
        in training."""
        mask_results = super().mask_loss(
            x=x,
            sampling_results=sampling_results,
            bbox_feats=bbox_feats,
            batch_gt_instances=batch_gt_instances)

        mask_point_results = self._mask_point_loss(
            x=x,
            sampling_results=sampling_results,
            mask_preds=mask_results['mask_preds'],
            batch_gt_instances=batch_gt_instances)
        mask_results['loss_mask'].update(
            loss_point=mask_point_results['loss_point'])

        return mask_results

    def _mask_point_loss(self, x: Tuple[Tensor],
                         sampling_results: List[SamplingResult],
                         mask_preds: Tensor,
                         batch_gt_instances: InstanceList) -> dict:
        """Run forward function and calculate loss for point head in
        training."""
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        rel_roi_points = self.point_head.get_roi_rel_points_train(
            mask_preds, pos_labels, cfg=self.train_cfg)
        rois = bbox2roi([res.pos_bboxes for res in sampling_results])

        fine_grained_point_feats = self._get_fine_grained_point_feats(
            x, rois, rel_roi_points)
        coarse_point_feats = point_sample(mask_preds, rel_roi_points)
        mask_point_pred = self.point_head(fine_grained_point_feats,
                                          coarse_point_feats)

        loss_and_target = self.point_head.loss_and_target(
            point_pred=mask_point_pred,
            rel_roi_points=rel_roi_points,
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            cfg=self.train_cfg)

        return loss_and_target

    def _mask_point_forward_test(self, x: Tuple[Tensor], rois: Tensor,
                                 label_preds: Tensor,
                                 mask_preds: Tensor) -> Tensor:
        """Mask refining process with point head in testing.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            rois (Tensor): shape (num_rois, 5).
            label_preds (Tensor): The predication class for each rois.
            mask_preds (Tensor): The predication coarse masks of
                shape (num_rois, num_classes, small_size, small_size).

        Returns:
            Tensor: The refined masks of shape (num_rois, num_classes,
            large_size, large_size).
        """
        refined_mask_pred = mask_preds.clone()
        for subdivision_step in range(self.test_cfg.subdivision_steps):
            refined_mask_pred = F.interpolate(
                refined_mask_pred,
                scale_factor=self.test_cfg.scale_factor,
                mode='bilinear',
                align_corners=False)
            # If `subdivision_num_points` is larger or equal to the
            # resolution of the next step, then we can skip this step
            num_rois, channels, mask_height, mask_width = \
                refined_mask_pred.shape
            if (self.test_cfg.subdivision_num_points >=
                    self.test_cfg.scale_factor**2 * mask_height * mask_width
                    and
                    subdivision_step < self.test_cfg.subdivision_steps - 1):
                continue
            point_indices, rel_roi_points = \
                self.point_head.get_roi_rel_points_test(
                    refined_mask_pred, label_preds, cfg=self.test_cfg)

            fine_grained_point_feats = self._get_fine_grained_point_feats(
                x=x, rois=rois, rel_roi_points=rel_roi_points)
            coarse_point_feats = point_sample(mask_preds, rel_roi_points)
            mask_point_pred = self.point_head(fine_grained_point_feats,
                                              coarse_point_feats)

            point_indices = point_indices.unsqueeze(1).expand(-1, channels, -1)
            refined_mask_pred = refined_mask_pred.reshape(
                num_rois, channels, mask_height * mask_width)
            refined_mask_pred = refined_mask_pred.scatter_(
                2, point_indices, mask_point_pred)
            refined_mask_pred = refined_mask_pred.view(num_rois, channels,
                                                       mask_height, mask_width)

        return refined_mask_pred

    def _get_fine_grained_point_feats(self, x: Tuple[Tensor], rois: Tensor,
                                      rel_roi_points: Tensor) -> Tensor:
        """Sample fine grained feats from each level feature map and
        concatenate them together.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            rois (Tensor): shape (num_rois, 5).
            rel_roi_points (Tensor): A tensor of shape (num_rois, num_points,
                2) that contains [0, 1] x [0, 1] normalized coordinates of the
                most uncertain points from the [mask_height, mask_width] grid.

        Returns:
            Tensor: The fine grained features for each points,
            has shape (num_rois, feats_channels, num_points).
        """
        assert rois.shape[0] > 0, 'RoI is a empty tensor.'
        num_imgs = x[0].shape[0]
        fine_grained_feats = []
        for idx in range(self.mask_roi_extractor.num_inputs):
            feats = x[idx]
            spatial_scale = 1. / float(
                self.mask_roi_extractor.featmap_strides[idx])
            point_feats = []
            for batch_ind in range(num_imgs):
                # unravel batch dim
                feat = feats[batch_ind].unsqueeze(0)
                inds = (rois[:, 0].long() == batch_ind)
                if inds.any():
                    rel_img_points = rel_roi_point_to_rel_img_point(
                        rois=rois[inds],
                        rel_roi_points=rel_roi_points[inds],
                        img=feat.shape[2:],
                        spatial_scale=spatial_scale).unsqueeze(0)
                    point_feat = point_sample(feat, rel_img_points)
                    point_feat = point_feat.squeeze(0).transpose(0, 1)
                    point_feats.append(point_feat)
            fine_grained_feats.append(torch.cat(point_feats, dim=0))
        return torch.cat(fine_grained_feats, dim=1)

    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
            - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        mask_results = self._mask_forward(x, mask_rois)
        mask_preds = mask_results['mask_preds']
        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

        # refine mask_preds
        mask_rois = mask_rois.split(num_mask_rois_per_img, 0)
        mask_preds_refined = []
        for i in range(len(batch_img_metas)):
            labels = results_list[i].labels
            x_i = [xx[[i]] for xx in x]
            mask_rois_i = mask_rois[i]
            mask_rois_i[:, 0] = 0
            mask_pred_i = self._mask_point_forward_test(
                x_i, mask_rois_i, labels, mask_preds[i])
            mask_preds_refined.append(mask_pred_i)

        # TODO: Handle the case where rescale is false
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds_refined,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        return results_list
