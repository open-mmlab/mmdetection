# Modified from https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend  # noqa

import torch
import torch.nn.functional as F
from mmcv.ops import point_sample, rel_roi_point_to_rel_img_point

from mmdet.core import bbox2roi, bbox_mapping, merge_aug_masks
from .. import builder
from ..builder import HEADS
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class PointRendRoIHead(StandardRoIHead):
    """`PointRend <https://arxiv.org/abs/1912.08193>`_."""

    def __init__(self, point_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.with_bbox and self.with_mask
        self.init_point_head(point_head)

    def init_point_head(self, point_head):
        """Initialize ``point_head``"""
        self.point_head = builder.build_head(point_head)

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head and point head
        in training."""
        mask_results = super()._mask_forward_train(x, sampling_results,
                                                   bbox_feats, gt_masks,
                                                   img_metas)
        if mask_results['loss_mask'] is not None:
            loss_point = self._mask_point_forward_train(
                x, sampling_results, mask_results['mask_pred'], gt_masks,
                img_metas)
            mask_results['loss_mask'].update(loss_point)

        return mask_results

    def _mask_point_forward_train(self, x, sampling_results, mask_pred,
                                  gt_masks, img_metas):
        """Run forward function and calculate loss for point head in
        training."""
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        rel_roi_points = self.point_head.get_roi_rel_points_train(
            mask_pred, pos_labels, cfg=self.train_cfg)
        rois = bbox2roi([res.pos_bboxes for res in sampling_results])

        fine_grained_point_feats = self._get_fine_grained_point_feats(
            x, rois, rel_roi_points, img_metas)
        coarse_point_feats = point_sample(mask_pred, rel_roi_points)
        mask_point_pred = self.point_head(fine_grained_point_feats,
                                          coarse_point_feats)
        mask_point_target = self.point_head.get_targets(
            rois, rel_roi_points, sampling_results, gt_masks, self.train_cfg)
        loss_mask_point = self.point_head.loss(mask_point_pred,
                                               mask_point_target, pos_labels)

        return loss_mask_point

    def _get_fine_grained_point_feats(self, x, rois, rel_roi_points,
                                      img_metas):
        """Sample fine grained feats from each level feature map and
        concatenate them together."""
        num_imgs = len(img_metas)
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
                        rois[inds], rel_roi_points[inds], feat.shape[2:],
                        spatial_scale).unsqueeze(0)
                    point_feat = point_sample(feat, rel_img_points)
                    point_feat = point_feat.squeeze(0).transpose(0, 1)
                    point_feats.append(point_feat)
            fine_grained_feats.append(torch.cat(point_feats, dim=0))
        return torch.cat(fine_grained_feats, dim=1)

    def _mask_point_forward_test(self, x, rois, label_pred, mask_pred,
                                 img_metas):
        """Mask refining process with point head in testing."""
        refined_mask_pred = mask_pred.clone()
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
                    refined_mask_pred, label_pred, cfg=self.test_cfg)
            fine_grained_point_feats = self._get_fine_grained_point_feats(
                x, rois, rel_roi_points, img_metas)
            coarse_point_feats = point_sample(mask_pred, rel_roi_points)
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

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Obtain mask prediction without augmentation."""
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [[[] for _ in range(self.mask_head.num_classes)]
                            for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            # split batch mask prediction back to each image
            mask_pred = mask_results['mask_pred']
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            mask_preds = mask_pred.split(num_mask_roi_per_img, 0)
            mask_rois = mask_rois.split(num_mask_roi_per_img, 0)

            # apply mask post-processing to each image individually
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                else:
                    x_i = [xx[[i]] for xx in x]
                    mask_rois_i = mask_rois[i]
                    mask_rois_i[:, 0] = 0  # TODO: remove this hack
                    mask_pred_i = self._mask_point_forward_test(
                        x_i, mask_rois_i, det_labels[i], mask_preds[i],
                        [img_metas])
                    segm_result = self.mask_head.get_seg_masks(
                        mask_pred_i, _bboxes[i], det_labels[i], self.test_cfg,
                        ori_shapes[i], scale_factors[i], rescale)
                    segm_results.append(segm_result)
        return segm_results

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        """Test for mask head with test time augmentation."""
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_results = self._mask_forward(x, mask_rois)
                mask_results['mask_pred'] = self._mask_point_forward_test(
                    x, mask_rois, det_labels, mask_results['mask_pred'],
                    img_metas)
                # convert to numpy array to save memory
                aug_masks.append(
                    mask_results['mask_pred'].sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas, self.test_cfg)

            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        return segm_result
