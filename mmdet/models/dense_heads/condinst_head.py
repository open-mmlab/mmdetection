# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Dict, Tuple
import numpy as np
from pyparsing import ParseExpression
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, kaiming_init
from mmengine.data import InstanceData
from mmengine.model import BaseModule, ModuleList
from torch import Tensor, conv2d

from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, OptConfigType, MultiConfig,
                         OptInstanceList, RangeType, OptMultiConfig, reduce_mean)
from ..layers import fast_nms
from ..utils import images_to_levels, multi_apply, select_single_mlvl
from ..utils.misc import empty_instances
from .base_mask_head import BaseMaskHead
from .fcos_head import FCOSHead

@MODELS.register_module()
class FCOSwithControllerHead(FCOSHead):
    
    def __init__(self,
                *args,
                 num_gen_params: int = 169,
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.num_gen_params = num_gen_params
        
        super().__init__(
            *args,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.controller = nn.Conv2d(self.in_channels, self.num_gen_params, 3, 1, padding=1)
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)

    def forward_single(self, x: Tensor, scale: Scale,
                       stride: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj:`mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness
            predictions of input feature maps.
        """
        cls_feat = x
        reg_feat = x
        controller_pred = []

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
            controller_pred.append[self.controller(reg_feat)]
        
        bbox_pred = self.conv_reg(reg_feat)
        
        
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        
        return cls_score, bbox_pred, centerness, controller_pred

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        controller_pred: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points,
                                                batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
        self._raw_positive_infos.update(controller_preds=controller_pred)
        # self._raw_positive_infos.update(controller_preds=controller_pred)
        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    def get_targets(
            self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            tuple: Targets of each level.

            - concat_lvl_labels (list[Tensor]): Labels of each level.
            - concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
            level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_targets_single(
            self, gt_instances: InstanceData, points: Tensor,
            regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor]:
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets: Tensor) -> Tensor:
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

#YOLACT
    def get_positive_infos(self) -> InstanceList:
        controller_pred = self._raw_positive_infos['controller_preds']
        return controller_pred
        # assert len(self._raw_positive_infos) > 0
        # sampling_results = self._raw_positive_infos['sampling_results']
        # num_imgs = len(sampling_results)

        # coeff_pred_list = []
        # for coeff_pred_per_level in self._raw_positive_infos['coeff_preds']:
        #     coeff_pred_per_level = \
        #         coeff_pred_per_level.permute(
        #             0, 2, 3, 1).reshape(num_imgs, -1, self.num_protos)
        #     coeff_pred_list.append(coeff_pred_per_level)
        # coeff_preds = torch.cat(coeff_pred_list, dim=1)

        # pos_info_list = []
        
        # for idx, sampling_result in enumerate(sampling_results):
        #     pos_info = InstanceData()
        #     coeff_preds_single = coeff_preds[idx]
        #     pos_info.pos_assigned_gt_inds = \
        #         sampling_result.pos_assigned_gt_inds
        #     pos_info.pos_inds = sampling_result.pos_inds
        #     pos_info.coeffs = coeff_preds_single[sampling_result.pos_inds]
        #     pos_info.bboxes = sampling_result.pos_gt_bboxes
        #     pos_info_list.append(pos_info)

        return pos_info_list
#YOLACT
    def predict_by_feat(self,
                        cls_scores,
                        bbox_preds,
                        centerness,
                        controller_pred,
                        batch_img_metas,
                        cfg=None,
                        rescale=True,
                        **kwargs):
        """Similar to func:``AnchorHead.get_bboxes``, but additionally
        processes coeff_preds.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            coeff_preds (list[Tensor]): Mask coefficients for each scale
                level with shape (N, num_anchors * num_protos, H, W)
            batch_img_metas (list[dict]): Batch image meta info.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - coeffs (Tensor): the predicted mask coefficients of
                  instance inside the corresponding box has a shape
                  (n, num_protos).
        """

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            coeff_pred_list = select_single_mlvl(coeff_preds, img_id)
            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                coeff_preds_list=coeff_pred_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale)
            result_list.append(results)
        return result_list
#YOLACT
    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                centernesses_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigType,
                                rescale: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results. Similar to func:``AnchorHead._predict_by_feat_single``,
        but additionally processes coeff_preds_list and uses fast NMS instead
        of traditional NMS.

        Args:
            cls_score_list (list[Tensor]): Box scores for a single scale level
                Has shape (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_priors * 4, H, W).
            coeff_preds_list (list[Tensor]): Mask coefficients for a single
                scale level with shape (num_priors * num_protos, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid,
                has shape (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - coeffs (Tensor): the predicted mask coefficients of
                  instance inside the corresponding box has a shape
                  (n, num_protos).
        """
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_priors)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_coeffs = []
        for cls_score, bbox_pred, coeff_pred, priors in \
                zip(cls_score_list, bbox_pred_list,
                    coeff_preds_list, mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            coeff_pred = coeff_pred.permute(1, 2,
                                            0).reshape(-1, self.num_protos)

            if 0 < nms_pre < scores.shape[0]:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                priors = priors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                coeff_pred = coeff_pred[topk_inds, :]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_coeffs.append(coeff_pred)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = torch.cat(mlvl_valid_priors)
        multi_bboxes = self.bbox_coder.decode(
            priors, bbox_pred, max_shape=img_shape)

        multi_scores = torch.cat(mlvl_scores)
        multi_coeffs = torch.cat(mlvl_coeffs)

        return self._bbox_post_process(
            multi_bboxes=multi_bboxes,
            multi_scores=multi_scores,
            multi_coeffs=multi_coeffs,
            cfg=cfg,
            rescale=rescale,
            img_meta=img_meta)
#YOLACT
    def _bbox_post_process(self,
                           multi_bboxes: Tensor,
                           multi_scores: Tensor,
                           multi_coeffs: Tensor,
                           cfg: ConfigType,
                           rescale: bool = False,
                           img_meta: Optional[dict] = None,
                           **kwargs) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            multi_bboxes (Tensor): Predicted bbox that concat all levels.
            multi_scores (Tensor): Bbox scores that concat all levels.
            multi_coeffs (Tensor): Mask coefficients  that concat all levels.
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - coeffs (Tensor): the predicted mask coefficients of
                  instance inside the corresponding box has a shape
                  (n, num_protos).
        """
        if rescale:
            assert img_meta.get('scale_factor') is not None
            multi_bboxes /= multi_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))
            # mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class

            padding = multi_scores.new_zeros(multi_scores.shape[0], 1)
            multi_scores = torch.cat([multi_scores, padding], dim=1)
        det_bboxes, det_labels, det_coeffs = fast_nms(
            multi_bboxes, multi_scores, multi_coeffs, cfg.score_thr,
            cfg.iou_thr, cfg.top_k, cfg.max_per_img)
        results = InstanceData()
        results.bboxes = det_bboxes[:, :4]
        results.scores = det_bboxes[:, -1]
        results.labels = det_labels
        results.coeffs = det_coeffs
        return results

@MODELS.register_module()
class CondInstDynamicMaskHead(BaseMaskHead):

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        in_features_channels:List[int] = [512, 1024, 2048],
        tower_channel: int = 128,
        num_convs: int = 3,
        num_outputs: int = 1,
        loss_mask_weight: float = 1.0,
        max_masks_to_train: int = 100,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        with_seg_branch: bool = True,
        boxinst_enabled: bool = False,
        loss_segm: ConfigType = dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        init_cfg=dict(
            type='Xavier',
            distribution='uniform',
            override=dict(name='protonet'))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.boxinst_enabled = boxinst_enabled
        self.in_features_channels = in_features_channels
        self.tower_channel = tower_channel
        # Segmentation branch
        self.with_seg_branch = with_seg_branch
        self.segm_branch = SegmentationModule(
            num_classes=num_classes, in_channels=in_channels) \
            if with_seg_branch else None
        self.loss_segm = MODELS.build(loss_segm) if with_seg_branch else None
        self.loss_mask_weight = loss_mask_weight
        self.num_classes = num_classes
        self.num_convs = num_convs
        self.num_outputs = num_outputs
        self.max_masks_to_train = max_masks_to_train
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        in_channels = self.in_channels
        # feature_channels = {k: v.channels for k, v in input_shape.items()}


        self.refine = nn.ModuleList()
        for in_feature in self.in_features_channels:
            self.refine.append(kaiming_init(nn.Conv2d(in_feature,self.tower_channel, 3, 1)))
            # self.refine.append((torch.conv2d(256,self.tower_channel, 3, 1)))
        self.tower = nn.ModuleList()
        for i in range(self.num_convs):
            self.tower.append(kaiming_init(nn.Conv2d(self.tower_channel, self.tower_channel, 3, 1)))
            # self.refine.append((torch.conv2d(256,self.tower_channel, 3, 1)))
        self.tower.append(nn.Conv2d(
            self.tower_channel, max(self.num_outputs, 1), 1
        ))

    def forward(self, x: tuple, positive_infos: InstanceList) -> tuple:
        pred_instances = positive_infos["instances"]
        if self.train:
            assert (self.max_proposals == -1) or (self.topk_proposals_per_im == -1), \
            "MAX_PROPOSALS and TOPK_PROPOSALS_PER_IM cannot be used at the same time."
            if self.max_proposals != -1:
                if self.max_proposals < len(pred_instances):
                    inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
                    pred_instances = pred_instances[inds[:self.max_proposals]]
            elif self.topk_proposals_per_im != -1:
                num_images = len(gt_instances)
                kept_instances = []
                for im_id in range(num_images):
                    instances_per_im = pred_instances[pred_instances.im_inds == im_id]
                    if len(instances_per_im) == 0:
                        kept_instances.append(instances_per_im)
                        continue
                    unique_gt_inds = instances_per_im.gt_inds.unique()
                    num_instances_per_gt = max(int(self.topk_proposals_per_im / len(unique_gt_inds)), 1)
                    for gt_ind in unique_gt_inds:
                        instances_per_gt = instances_per_im[instances_per_im.gt_inds == gt_ind]
                        if len(instances_per_gt) > num_instances_per_gt:
                            scores = instances_per_gt.logits_pred.sigmoid().max(dim=1)[0]
                            ctrness_pred = instances_per_gt.ctrness_pred.sigmoid()
                            inds = (scores * ctrness_pred).topk(k=num_instances_per_gt, dim=0)[1]
                            instances_per_gt = instances_per_gt[inds]
                        kept_instances.append(instances_per_gt)
                pred_instances = Instances.cat(kept_instances)
            pred_instances.mask_head_params = pred_instances.top_feats
        seg_x = x[1]
        mask_x = x[1:]
        mask_feat_stride = 8
        if self.with_seg_branch  is not None and self.training:
            segm_preds = self.segm_branch(seg_x)
        else:
            segm_preds = None
        
        mask_pred_list = []

        for i, f in enumerate(mask_x):
            if i == 0:
                mask_x_ori = self.refine[i](f)
            else:
                mask_x_p = self.refine[i](f)

                target_h, target_w = mask_x_ori.size()[2:]
                h, w = mask_x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                mask_x_p = self.aligned_bilinear(mask_x_p, factor_h)
                mask_x_ori = mask_x_ori + mask_x_p

        mask_feats = self.tower(mask_x_ori)

        if self.num_outputs == 0:
            mask_feats = mask_feats[:, :self.num_outputs]
        
        locations = self.compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(pred_instances)

        im_inds = pred_instances.im_inds
        mask_head_params = positive_infos["controller"]

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = pred_instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[pred_instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = self._parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        n_layers = len(weights)
        for i, (w, b) in enumerate(zip(weights, biases)):
            mask_head_inputs = F.conv2d(
                mask_head_inputs, w, bias=b,
                stride=1, padding=0,
                groups=n_inst
            )
            if i < n_layers - 1:
                mask_head_inputs = F.relu(mask_head_inputs)
        mask_logits = mask_head_inputs 
        mask_logits = mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = self.aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))
        mask_preds = torch.sigmoid(mask_logits)
        mask_pred_list.append(mask_preds)
        return mask_pred_list, segm_preds

    def loss_by_feat(self, mask_preds: List[Tensor], segm_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict], positive_infos: InstanceList,
                     **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the mask head.

        Args:
            mask_preds (list[Tensor]): List of predicted prototypes, each has
                shape (num_classes, H, W).
            segm_preds (Tensor):  Predicted semantic segmentation map with
                shape (N, num_classes, H, W)
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``masks``,
                and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of multiple images.
            positive_infos (List[:obj:``InstanceData``]): Information of
                positive samples of each image that are assigned in detection
                head.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert positive_infos is not None, \
            'positive_infos should not be None in `YOLACTProtonet`'
        losses = dict()

        # crop
        croped_mask_pred = self.crop_mask_preds(mask_preds, batch_img_metas,
                                                positive_infos)

        loss_mask = []
        loss_segm = []
        num_imgs, _, mask_h, mask_w = segm_preds.size()
        assert num_imgs == len(croped_mask_pred)
        segm_avg_factor = num_imgs * mask_h * mask_w
        total_pos = 0

        if self.segm_branch is not None:
            assert segm_preds is not None

        for idx in range(num_imgs):
            img_meta = batch_img_metas[idx]

            (mask_preds, pos_mask_targets, segm_targets, num_pos,
            gt_bboxes_for_reweight) = self._get_targets_single(
                croped_mask_pred[idx], segm_preds[idx],
                batch_gt_instances[idx], positive_infos[idx])

            # segmentation loss
            if self.with_seg_branch:
                if segm_targets is None:
                    loss = segm_preds[idx].sum() * 0.
                else:
                    loss = self.loss_segm(
                        segm_preds[idx],
                        segm_targets,
                        avg_factor=segm_avg_factor)
                loss_segm.append(loss)
            
            # mask loss
            
            total_pos += num_pos
            if num_pos == 0 or pos_mask_targets is None:
                loss = mask_preds.sum() * 0.
                if self.boxinst_enabled:
                    pass
            else:
                #Condinst
                if self.boxinst_enabled:
                    pass
                else:
                    # gt_inds = pred_instances.gt_inds
                    # gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
                    # gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

                    eps = 1e-5
                    n_inst = mask_preds.size(0)
                    mask_preds = mask_preds.reshape(n_inst, -1)
                    gt_bitmasks = gt_bitmasks.reshape(n_inst, -1)
                    intersection = (mask_preds * gt_bitmasks).sum(dim=1)
                    union = (mask_preds ** 2.0).sum(dim=1) + (gt_bitmasks ** 2.0).sum(dim=1) + eps
                    loss = 1. - (2 * intersection / union).mean()
            
            loss_mask.append(loss)

        if total_pos == 0:
            total_pos += 1  # avoid nan
        loss_mask = [x / total_pos for x in loss_mask]

        losses.update(loss_mask=loss_mask)
        if self.with_seg_branch:
            losses.update(loss_segm=loss_segm)

        return losses

    def _get_targets_single(self, mask_preds: Tensor, segm_pred: Tensor,
                            gt_instances: InstanceData,
                            positive_info: InstanceData):
        """Compute targets for predictions of single image.

        Args:
            mask_preds (Tensor): Predicted prototypes with shape
                (num_classes, H, W).
            segm_pred (Tensor): Predicted semantic segmentation map
                with shape (num_classes, H, W).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes``, ``labels``,
                and ``masks`` attributes.
            positive_info (:obj:`InstanceData`): Information of positive
                samples that are assigned in detection head. It usually
                contains following keys.

                    - pos_assigned_gt_inds (Tensor): Assigner GT indexes of
                      positive proposals, has shape (num_pos, )
                    - pos_inds (Tensor): Positive index of image, has
                      shape (num_pos, ).
                    - coeffs (Tensor): Positive mask coefficients
                      with shape (num_pos, num_protos).
                    - bboxes (Tensor): Positive bboxes with shape
                      (num_pos, 4)

        Returns:
            tuple: Usually returns a tuple containing learning targets.

            - mask_preds (Tensor): Positive predicted mask with shape
              (num_pos, mask_h, mask_w).
            - pos_mask_targets (Tensor): Positive mask targets with shape
              (num_pos, mask_h, mask_w).
            - segm_targets (Tensor): Semantic segmentation targets with shape
              (num_classes, segm_h, segm_w).
            - num_pos (int): Positive numbers.
            - gt_bboxes_for_reweight (Tensor): GT bboxes that match to the
              positive priors has shape (num_pos, 4).
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        device = gt_bboxes.device
        gt_masks = gt_instances.masks.to_tensor(
            dtype=torch.bool, device=device).float()
        if gt_masks.size(0) == 0:
            return mask_preds, None, None, 0, None

        # process with semantic segmentation targets
        if segm_pred is not None:
            num_classes, segm_h, segm_w = segm_pred.size()
            with torch.no_grad():
                downsampled_masks = F.interpolate(
                    gt_masks.unsqueeze(0), (segm_h, segm_w),
                    mode='bilinear',
                    align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()
                segm_targets = torch.zeros_like(segm_pred, requires_grad=False)
                for obj_idx in range(downsampled_masks.size(0)):
                    segm_targets[gt_labels[obj_idx] - 1] = torch.max(
                        segm_targets[gt_labels[obj_idx] - 1],
                        downsampled_masks[obj_idx])
        else:
            segm_targets = None
        # process with mask targets
        pos_assigned_gt_inds = positive_info.pos_assigned_gt_inds
        num_pos = pos_assigned_gt_inds.size(0)
        # Since we're producing (near) full image masks,
        # it'd take too much vram to backprop on every single mask.
        # Thus we select only a subset.
        if num_pos > self.max_masks_to_train:
            perm = torch.randperm(num_pos)
            select = perm[:self.max_masks_to_train]
            mask_preds = mask_preds[select]
            pos_assigned_gt_inds = pos_assigned_gt_inds[select]
            num_pos = self.max_masks_to_train

        gt_bboxes_for_reweight = gt_bboxes[pos_assigned_gt_inds]

        mask_h, mask_w = mask_preds.shape[-2:]
        gt_masks = F.interpolate(
            gt_masks.unsqueeze(0), (mask_h, mask_w),
            mode='bilinear',
            align_corners=False).squeeze(0)
        gt_masks = gt_masks.gt(0.5).float()
        pos_mask_targets = gt_masks[pos_assigned_gt_inds]

        return (mask_preds, pos_mask_targets, segm_targets, num_pos,
                gt_bboxes_for_reweight)

    def crop_mask_preds(self, mask_preds: List[Tensor],
                        batch_img_metas: List[dict],
                        positive_infos: InstanceList) -> list:
        """Crop predicted masks by zeroing out everything not in the predicted
        bbox.

        Args:
            mask_preds (list[Tensor]): Predicted prototypes with shape
                (num_classes, H, W).
            batch_img_metas (list[dict]): Meta information of multiple images.
            positive_infos (List[:obj:``InstanceData``]): Positive
                information that calculate from detect head.

        Returns:
            list: The cropped masks.
        """
        croped_mask_preds = []
        for img_meta, mask_preds, cur_info in zip(batch_img_metas, mask_preds,
                                                  positive_infos):
            bboxes_for_cropping = copy.deepcopy(cur_info.bboxes)
            h, w = img_meta['img_shape'][:2]
            bboxes_for_cropping[:, 0::2] /= w
            bboxes_for_cropping[:, 1::2] /= h
            mask_preds = self.crop_single(mask_preds, bboxes_for_cropping)
            mask_preds = mask_preds.permute(2, 0, 1).contiguous()
            croped_mask_preds.append(mask_preds)
        return croped_mask_preds

    def crop_single(self,
                    masks: Tensor,
                    boxes: Tensor,
                    padding: int = 1) -> Tensor:
        """Crop single predicted masks by zeroing out everything not in the
        predicted bbox.

        Args:
            masks (Tensor): Predicted prototypes, has shape [H, W, N].
            boxes (Tensor): Bbox coords in relative point form with
                shape [N, 4].
            padding (int): Image padding size.

        Return:
            Tensor: The cropped masks.
        """
        h, w, n = masks.size()
        x1, x2 = self.sanitize_coordinates(
            boxes[:, 0], boxes[:, 2], w, padding, cast=False)
        y1, y2 = self.sanitize_coordinates(
            boxes[:, 1], boxes[:, 3], h, padding, cast=False)

        rows = torch.arange(
            w, device=masks.device, dtype=x1.dtype).view(1, -1,
                                                         1).expand(h, w, n)
        cols = torch.arange(
            h, device=masks.device, dtype=x1.dtype).view(-1, 1,
                                                         1).expand(h, w, n)

        masks_left = rows >= x1.view(1, 1, -1)
        masks_right = rows < x2.view(1, 1, -1)
        masks_up = cols >= y1.view(1, 1, -1)
        masks_down = cols < y2.view(1, 1, -1)

        crop_mask = masks_left * masks_right * masks_up * masks_down

        return masks * crop_mask.float()

    def sanitize_coordinates(self,
                             x1: Tensor,
                             x2: Tensor,
                             img_size: int,
                             padding: int = 0,
                             cast: bool = True) -> tuple:
        """Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0,
        and x2 <= image_size. Also converts from relative to absolute
        coordinates and casts the results to long tensors.

        Warning: this does things in-place behind the scenes so
        copy if necessary.

        Args:
            x1 (Tensor): shape (N, ).
            x2 (Tensor): shape (N, ).
            img_size (int): Size of the input image.
            padding (int): x1 >= padding, x2 <= image_size-padding.
            cast (bool): If cast is false, the result won't be cast to longs.

        Returns:
            tuple:

            - x1 (Tensor): Sanitized _x1.
            - x2 (Tensor): Sanitized _x2.
        """
        x1 = x1 * img_size
        x2 = x2 * img_size
        if cast:
            x1 = x1.long()
            x2 = x2.long()
        x1 = torch.min(x1, x2)
        x2 = torch.max(x1, x2)
        x1 = torch.clamp(x1 - padding, min=0)
        x2 = torch.clamp(x2 + padding, max=img_size)
        return x1, x2

    def predict_by_feat(self,
                        mask_preds: List[Tensor],
                        segm_preds: Tensor,
                        results_list: InstanceList,
                        batch_img_metas: List[dict],
                        rescale: bool = True,
                        **kwargs) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        mask results.

        Args:
            mask_preds (list[Tensor]): Predicted prototypes with shape
                (num_classes, H, W).
            results_list (List[:obj:``InstanceData``]): BBoxHead results.
            batch_img_metas (list[dict]): Meta information of all images.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Processed results of multiple
            images.Each :obj:`InstanceData` usually contains
            following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """
        assert len(mask_preds) == len(results_list) == len(batch_img_metas)

        croped_mask_pred = self.crop_mask_preds(mask_preds, batch_img_metas,
                                                results_list)

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = results_list[img_id]
            bboxes = results.bboxes
            mask_preds = croped_mask_pred[img_id]
            if bboxes.shape[0] == 0 or mask_preds.shape[0] == 0:
                results_list[img_id] = empty_instances(
                    [img_meta],
                    bboxes.device,
                    task_type='mask',
                    instance_results=[results])[0]
            else:
                im_mask = self._predict_by_feat_single(
                    mask_preds=croped_mask_pred[img_id],
                    bboxes=bboxes,
                    img_meta=img_meta,
                    rescale=rescale)
                results.masks = im_mask
        return results_list

    def _predict_by_feat_single(self,
                                mask_preds: Tensor,
                                bboxes: Tensor,
                                img_meta: dict,
                                rescale: bool,
                                cfg: OptConfigType = None):
        """Transform a single image's features extracted from the head into
        mask results.

        Args:
            mask_preds (Tensor): Predicted prototypes, has shape [H, W, N].
            bboxes (Tensor): Bbox coords in relative point form with
                shape [N, 4].
            img_meta (dict): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If rescale is False, then returned masks will
                fit the scale of imgs[0].
            cfg (dict, optional): Config used in test phase.
                Defaults to None.

        Returns:
            :obj:`InstanceData`: Processed results of single image.
             it usually contains following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """
        
        cfg = self.test_cfg if cfg is None else cfg
        scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            (1, 2))
        img_h, img_w = img_meta['ori_shape'][:2]
        if rescale:  # in-placed rescale the bboxes
            scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
                (1, 2))
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)

        masks = F.interpolate(
            mask_preds.unsqueeze(0), (img_h, img_w),
            mode='bilinear',
            align_corners=False).squeeze(0) > cfg.mask_thr

        if cfg.mask_thr_binary < 0:
            # for visualization and debugging
            masks = (masks * 255).to(dtype=torch.uint8)
        return masks

    def _parse_dynamic_params(params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts)

        return weight_splits, bias_splits
    
    def aligned_bilinear(tensor, factor):
        assert tensor.dim() == 4
        assert factor >= 1
        assert int(factor) == factor

        if factor == 1:
            return tensor

        h, w = tensor.size()[2:]
        tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
        oh = factor * h + 1
        ow = factor * w + 1
        tensor = F.interpolate(
            tensor, size=(oh, ow),
            mode='bilinear',
            align_corners=True
        )
        tensor = F.pad(
            tensor, pad=(factor // 2, 0, factor // 2, 0),
            mode="replicate"
        )

        return tensor[:, :, :oh - 1, :ow - 1]

    def compute_locations(h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

class SegmentationModule(BaseModule):

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        init_cfg: ConfigType = dict(
            type='Xavier',
            distribution='uniform',
            override=dict(name='segm_conv'))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.segm_conv = nn.Conv2d(
            self.in_channels, self.num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward feature from the upstream network.

        Args:
            x (Tensor): Feature from the upstream network, which is
                a 4D-tensor.

        Returns:
            Tensor: Predicted semantic segmentation map with shape
                (N, num_classes, H, W).
        """
        return self.segm_conv(x)
