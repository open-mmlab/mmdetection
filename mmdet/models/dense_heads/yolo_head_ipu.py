# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import warnings

import torch
from mmcv.runner import force_fp32
from mmcv.utils import IS_IPU_AVAILABLE

if IS_IPU_AVAILABLE:
    from mmcv.device.ipu import nms_ipu, remap_tensor, slice_statically

from mmdet.core import multi_apply
from ..builder import HEADS
from .yolo_head import YOLOV3Head


@HEADS.register_module()
class IPUYOLOV3Head(YOLOV3Head):
    """Implemented YOLOV3Head in a static way to run on IPU.

    Args are same as YOLOV3Head
    """

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        outs = self.forward(feats)
        results_list = self.get_bboxes_statically(
            *outs, img_metas=img_metas, rescale=rescale)
        return results_list

    @force_fp32(apply_to=('pred_maps', ))
    def get_bboxes_statically(self,
                              pred_maps,
                              img_metas,
                              cfg=None,
                              rescale=False,
                              with_nms=True):
        """Transform network output for a batch into bbox predictions. It is an
        IPU static version.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert with_nms, 'NMS is hardcoded in the IPU implementation'
        # There is no fp16 in the trace phase,
        # so force_fp32 does not take effect
        pred_maps = [ele.float() for ele in pred_maps]

        assert len(pred_maps) == self.num_levels
        cfg = self.test_cfg if cfg is None else cfg

        num_imgs = len(img_metas)
        featmap_sizes = [pred_map.shape[-2:] for pred_map in pred_maps]

        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=pred_maps[0].device)
        flatten_preds = []
        flatten_strides = []
        for pred, stride in zip(pred_maps, self.featmap_strides):
            pred = pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.num_attrib)
            pred_logits = torch.sigmoid(pred[..., :2])
            pred = torch.cat([pred_logits, pred[..., 2:]], dim=-1)
            flatten_preds.append(pred)
            flatten_strides.append(
                pred.new_tensor(stride).expand(pred.size(1)))

        flatten_preds = torch.cat(flatten_preds, dim=1)
        flatten_bbox_preds = flatten_preds[..., :4]
        flatten_objectness = flatten_preds[..., 4].sigmoid()
        flatten_cls_scores = flatten_preds[..., 5:].sigmoid()
        flatten_anchors = torch.cat(mlvl_anchors)
        flatten_strides = torch.cat(flatten_strides)
        flatten_bboxes = self.bbox_coder.decode(flatten_anchors,
                                                flatten_bbox_preds,
                                                flatten_strides.unsqueeze(-1))

        if rescale:
            scale_factors = []
            for img_meta in img_metas:
                scale_factor = img_meta['scale_factor'].unsqueeze(0)
                scale_factor = scale_factor.unsqueeze(0)
                scale_factors.append(scale_factor)
            scale_factors = torch.cat(scale_factors, dim=0)
            flatten_bboxes = flatten_bboxes / scale_factors

        padding = flatten_bboxes.new_zeros(num_imgs, flatten_bboxes.shape[1],
                                           1)
        flatten_cls_scores = torch.cat([flatten_cls_scores, padding], dim=-1)

        det_results = []
        for (bboxes, scores, objectness) in zip(flatten_bboxes,
                                                flatten_cls_scores,
                                                flatten_objectness):
            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            if conf_thr > 0:
                conf_inds = objectness >= conf_thr
                bboxes = bboxes.float() * conf_inds.unsqueeze(-1)
                scores = scores.float() * conf_inds.unsqueeze(-1)
                objectness = objectness.float() * conf_inds

            # last column is background
            num_class_exclude_groud = scores.shape[-1] - 1
            det_bboxes_list = []
            det_labels_list = []
            for i in range(num_class_exclude_groud):
                mask_by_score_thr = scores[:, i] > cfg.score_thr
                single_class_score = scores[:, i] * objectness
                single_class_bboxes = bboxes.clone() * \
                    mask_by_score_thr.unsqueeze(-1)
                single_class_score = single_class_score * mask_by_score_thr
                iou_thrd = cfg.nms['iou_threshold']
                num_dets = cfg.max_per_img
                single_class_bboxes = remap_tensor(single_class_bboxes)
                single_class_score = remap_tensor(single_class_score)
                _, det_boxes, boxes_keep = nms_ipu(single_class_bboxes,
                                                   single_class_score,
                                                   iou_thrd, num_dets)
                det_scores = slice_statically(
                    single_class_score, boxes_keep, dim=0).unsqueeze(-1)
                valid_flags = (boxes_keep >= 0).unsqueeze(-1)
                det_scores = det_scores * valid_flags
                det_bboxes = torch.cat([det_boxes, det_scores], dim=1)
                det_labels = torch.ones([det_bboxes.shape[0]],
                                        dtype=torch.long) * i
                det_bboxes_list.append(det_bboxes)
                det_labels_list.append(det_labels)
            _det_bboxes = torch.cat(det_bboxes_list, dim=0)
            _det_labels = torch.cat(det_labels_list, dim=0)

            det_scores = _det_bboxes[:, 4]
            keep = torch.topk(det_scores, k=cfg.max_per_img).indices
            det_bboxes = slice_statically(_det_bboxes, keep, dim=0)
            det_labels = slice_statically(_det_labels, keep, dim=0)
            det_results.append(tuple([det_bboxes, det_labels]))
        return det_results

    @force_fp32(apply_to=('pred_maps', ))
    def loss(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             target_maps_list=None,
             neg_maps_list=None):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            target_map_list (list[Tensor]): Target map of each level.
            neg_map_list (list[Tensor]): Negative map of each level.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if target_maps_list is None or neg_maps_list is None:
            num_imgs = len(img_metas)
            device = pred_maps[0][0].device

            featmap_sizes = [
                pred_maps[i].shape[-2:] for i in range(self.num_levels)
            ]
            mlvl_anchors = self.prior_generator.grid_priors(
                featmap_sizes, device=device)
            anchor_list = [mlvl_anchors for _ in range(num_imgs)]

            responsible_flag_list = []
            for img_id in range(len(img_metas)):
                responsible_flag_list.append(
                    self.prior_generator.responsible_flags(
                        featmap_sizes, gt_bboxes[img_id], device))

            target_maps_list, neg_maps_list = self.get_targets(
                anchor_list, responsible_flag_list, gt_bboxes, gt_labels)

        losses_cls, losses_conf, losses_xy, losses_wh = multi_apply(
            self.loss_single, pred_maps, target_maps_list, neg_maps_list)

        return dict(
            loss_cls=losses_cls,
            loss_conf=losses_conf,
            loss_xy=losses_xy,
            loss_wh=losses_wh)

    def loss_single(self, pred_map, target_map, neg_map):
        """Compute loss of a single image from a batch.

        Args:
            pred_map (Tensor): Raw predictions for a single level.
            target_map (Tensor): The Ground-Truth target for a single level.
            neg_map (Tensor): The negative masks for a single level.

        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """

        num_imgs = len(pred_map)
        pred_map = pred_map.permute(0, 2, 3,
                                    1).reshape(num_imgs, -1, self.num_attrib)
        neg_mask = neg_map.float()
        pos_mask = target_map[..., 4]
        pos_and_neg_mask = neg_mask + pos_mask
        pos_mask = pos_mask.unsqueeze(dim=-1)
        if torch.max(pos_and_neg_mask) > 1.:
            warnings.warn('There is overlap between pos and neg sample.')
            pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)

        pred_xy = pred_map[..., :2]
        pred_wh = pred_map[..., 2:4]
        pred_conf = pred_map[..., 4]
        pred_label = pred_map[..., 5:]

        target_xy = target_map[..., :2]
        target_wh = target_map[..., 2:4]
        target_conf = target_map[..., 4]
        target_label = target_map[..., 5:]

        pred_xy = remap_tensor(pred_xy)
        pred_wh = remap_tensor(pred_wh)
        pred_conf = remap_tensor(pred_conf)
        pred_label = remap_tensor(pred_label)
        target_xy = remap_tensor(target_xy)
        target_wh = remap_tensor(target_wh)
        target_conf = remap_tensor(target_conf)
        target_label = remap_tensor(target_label)

        loss_cls = self.loss_cls(pred_label, target_label, weight=pos_mask)
        loss_conf = self.loss_conf(
            pred_conf, target_conf, weight=pos_and_neg_mask)
        loss_xy = self.loss_xy(pred_xy, target_xy, weight=pos_mask)
        loss_wh = self.loss_wh(pred_wh, target_wh, weight=pos_mask)

        return loss_cls, loss_conf, loss_xy, loss_wh
