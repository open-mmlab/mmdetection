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

INF = 100000000

@MODELS.register_module()
class FCOSwithControllerHead(FCOSHead):
    
    def __init__(self,
                *args,
                 num_gen_params: int = 169,
                 in_features: List = ["p3", "p4", "p5", "p6", "p7"],
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
        self.in_features = in_features
        
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
            controller_pred.append(self.controller(reg_feat))
        
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

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.sub_compute_locations(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def sub_compute_locations(self, h, w, stride, device):
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

        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        
        locations = torch.cat(locations, dim=0)
        locations = [locations.clone() for _ in range(len(batch_gt_instances))]

        fpn_levels = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(locations)
        ]

        self._raw_positive_infos.update(controller_preds=controller_pred)
        self._raw_positive_infos.update(locations=locations)
        self._raw_positive_infos.update(fpn_levels=fpn_levels)

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
        self._raw_positive_infos.update(instances=concat_lvl_bbox_targets)

        return concat_lvl_labels, concat_lvl_bbox_targets

    def get_positive_infos(self) -> InstanceList:
        assert len(self._raw_positive_infos) > 0
        controller_preds = self._raw_positive_infos['controller_preds']
        sampling_results = self._raw_positive_infos['instances']
        return [controller_preds, sampling_results]

@MODELS.register_module()
class CondInstDynamicMaskHead(BaseMaskHead):

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        in_features_channels:List[int] = [256, 256, 256],
        tower_channel: int = 128,
        num_convs: int = 3,
        num_outputs: int = 8,
        max_proposals: int = 100,
        topk_proposals_per_im: int = -1,
        loss_mask_weight: float = 1.0,
        max_masks_to_train: int = 100,
        disable_rel_coords: bool = False,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        with_seg_branch: bool = True,
        boxinst_enabled: bool = False,
        sizes_of_interest: List[int] = [64, 128, 256, 512],
        loss_segm: ConfigType = dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        init_cfg = dict(
            type='Xavier',
            distribution='uniform',
        )
        # init_cfg=dict(
        #     type='Xavier',
        #     distribution='uniform',
        #     override=dict(name='protonet'))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.disable_rel_coords = disable_rel_coords
        self.in_channels = in_channels
        self.boxinst_enabled = boxinst_enabled
        self.in_features_channels = in_features_channels
        self.tower_channel = tower_channel
        # Segmentation branch
        self.with_seg_branch = with_seg_branch
        # self.segm_branch = nn.Conv2d(
        #     self.in_channels, self.num_classes, kernel_size=1)
        self.segm_branch = SegmentationModule(
            num_classes=num_classes, in_channels=in_channels) \
            if with_seg_branch else None
        self.loss_segm = MODELS.build(loss_segm) if with_seg_branch else None
        self.loss_mask_weight = loss_mask_weight
        self.num_classes = num_classes
        self.num_convs = num_convs
        self.num_outputs = num_outputs
        self.max_proposals = max_proposals
        self.topk_proposals_per_im = topk_proposals_per_im
        self.max_masks_to_train = max_masks_to_train
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.soi = sizes_of_interest
        self.sizes_of_interest = torch.tensor(self.soi + [self.soi[-1] * 2])

        self._init_layers()

    def _init_layers(self) -> None:
        in_channels = self.in_channels
        # feature_channels = {k: v.channels for k, v in input_shape.items()}

        self.refine = nn.ModuleList()
        for in_feature in self.in_features_channels:
            self.refine.append(nn.Conv2d(in_feature,self.tower_channel, 3, 1, 1))
        tower = []
        for i in range(self.num_convs):
            tower.append(nn.Conv2d(self.tower_channel, self.tower_channel, 3, 1, 1))
        tower.append(nn.Conv2d(
            self.tower_channel, max(self.num_outputs, 1), 1
        ))
        self.add_module('tower', nn.Sequential(*tower))

    def forward(self, x: tuple, positive_infos: InstanceList) -> tuple:
        
        pred_instances = positive_infos[1]

        # if self.train:
        #     assert (self.max_proposals == -1) or (self.topk_proposals_per_im == -1), \
        #     "MAX_PROPOSALS and TOPK_PROPOSALS_PER_IM cannot be used at the same time."
        #     if self.max_proposals != -1:
        #         if self.max_proposals < len(pred_instances):
        #             inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
        #             pred_instances = pred_instances[inds[:self.max_proposals]]
        #     elif self.topk_proposals_per_im != -1:
        #         num_images = len(pred_instances)
        #         kept_instances = []
        #         for im_id in range(num_images):
        #             instances_per_im = pred_instances[pred_instances.im_inds == im_id]
        #             if len(instances_per_im) == 0:
        #                 kept_instances.append(instances_per_im)
        #                 continue
        #             unique_gt_inds = instances_per_im.gt_inds.unique()
        #             num_instances_per_gt = max(int(self.topk_proposals_per_im / len(unique_gt_inds)), 1)
        #             for gt_ind in unique_gt_inds:
        #                 instances_per_gt = instances_per_im[instances_per_im.gt_inds == gt_ind]
        #                 if len(instances_per_gt) > num_instances_per_gt:
        #                     scores = instances_per_gt.logits_pred.sigmoid().max(dim=1)[0]
        #                     ctrness_pred = instances_per_gt.ctrness_pred.sigmoid()
        #                     inds = (scores * ctrness_pred).topk(k=num_instances_per_gt, dim=0)[1]
        #                     instances_per_gt = instances_per_gt[inds]
        #                 kept_instances.append(instances_per_gt)
        #         pred_instances = Instances.cat(kept_instances)
        #     pred_instances.mask_head_params = pred_instances.top_feats
        
        seg_x = x[1]
        mask_x = x[2:]
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

        # im_inds = pred_instances.im_inds
        im_inds = [
            locations.new_ones(locations.size(0), dtype=torch.long) * i for i in range(len(pred_instances))
        ]
        # TODO

        mask_head_params = positive_infos[0]

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            # TODO
            # instance_locations = pred_instances.locations
            instance_locations = locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            # soi = self.sizes_of_interest.float()[pred_instances.fpn_levels]

            fpn_levels = [
                loc.new_ones(len(loc), dtype=torch.long) * level
                for level, loc in enumerate(locations)
            ]

            soi = self.sizes_of_interest.float()[fpn_levels]
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

    def _parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
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
    
    def aligned_bilinear(self, tensor, factor):
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

    def compute_locations(self, h, w, stride, device):
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
