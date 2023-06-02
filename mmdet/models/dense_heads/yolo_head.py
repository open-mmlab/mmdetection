# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import copy
import warnings
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, is_norm
from mmengine.model import bias_init_with_prob, constant_init, normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList)
from ..task_modules.samplers import PseudoSampler
from ..utils import filter_scores_and_topk, images_to_levels, multi_apply
from .base_dense_head import BaseDenseHead


@MODELS.register_module()
class YOLOV3Head(BaseDenseHead):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (Sequence[int]): Number of input channels per scale.
        out_channels (Sequence[int]): The number of output channels per scale
            before the final 1x1 layer. Default: (1024, 512, 256).
        anchor_generator (:obj:`ConfigDict` or dict): Config dict for anchor
            generator.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bounding box coder.
        featmap_strides (Sequence[int]): The stride of each scale.
            Should be in descending order. Defaults to (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Defaults to 0.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. Defaults to dict(type='BN', requires_grad=True).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='LeakyReLU', negative_slope=0.1).
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_conf (:obj:`ConfigDict` or dict): Config of confidence loss.
        loss_xy (:obj:`ConfigDict` or dict): Config of xy coordinate loss.
        loss_wh (:obj:`ConfigDict` or dict): Config of wh coordinate loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            YOLOV3 head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            YOLOV3 head. Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Sequence[int],
                 out_channels: Sequence[int] = (1024, 512, 256),
                 anchor_generator: ConfigType = dict(
                     type='YOLOAnchorGenerator',
                     base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(10, 13), (16, 30), (33, 23)]],
                     strides=[32, 16, 8]),
                 bbox_coder: ConfigType = dict(type='YOLOBBoxCoder'),
                 featmap_strides: Sequence[int] = (32, 16, 8),
                 one_hot_smoother: float = 0.,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: ConfigType = dict(
                     type='LeakyReLU', negative_slope=0.1),
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_conf: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_xy: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_wh: ConfigType = dict(type='MSELoss', loss_weight=1.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=None)
        # Check params
        assert (len(in_channels) == len(out_channels) == len(featmap_strides))

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            if train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'], context=self)
            else:
                self.sampler = PseudoSampler()

        self.one_hot_smoother = one_hot_smoother

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.bbox_coder = TASK_UTILS.build(bbox_coder)

        self.prior_generator = TASK_UTILS.build(anchor_generator)

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_conf = MODELS.build(loss_conf)
        self.loss_xy = MODELS.build(loss_xy)
        self.loss_wh = MODELS.build(loss_wh)

        self.num_base_priors = self.prior_generator.num_base_priors[0]
        assert len(
            self.prior_generator.num_base_priors) == len(featmap_strides)
        self._init_layers()

    @property
    def num_levels(self) -> int:
        """int: number of feature map levels"""
        return len(self.featmap_strides)

    @property
    def num_attrib(self) -> int:
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv3 head."""
        self.convs_bridge = nn.ModuleList()
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_bridge = ConvModule(
                self.in_channels[i],
                self.out_channels[i],
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            conv_pred = nn.Conv2d(self.out_channels[i],
                                  self.num_base_priors * self.num_attrib, 1)

            self.convs_bridge.append(conv_bridge)
            self.convs_pred.append(conv_pred)

    def init_weights(self) -> None:
        """initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

        # Use prior in model initialization to improve stability
        for conv_pred, stride in zip(self.convs_pred, self.featmap_strides):
            bias = conv_pred.bias.reshape(self.num_base_priors, -1)
            # init objectness with prior of 8 objects per feature map
            # refer to https://github.com/ultralytics/yolov3
            nn.init.constant_(bias.data[:, 4],
                              bias_init_with_prob(8 / (608 / stride)**2))
            nn.init.constant_(bias.data[:, 5:], bias_init_with_prob(0.01))

    def forward(self, x: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        assert len(x) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            feat = x[i]
            feat = self.convs_bridge[i](feat)
            pred_map = self.convs_pred[i](feat)
            pred_maps.append(pred_map)

        return tuple(pred_maps),

    def predict_by_feat(self,
                        pred_maps: Sequence[Tensor],
                        batch_img_metas: Optional[List[dict]],
                        cfg: OptConfigType = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results. It has been accelerated since PR #5991.

        Args:
            pred_maps (Sequence[Tensor]): Raw predictions for a batch of
                images.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (:obj:`ConfigDict` or dict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
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
        """
        assert len(pred_maps) == self.num_levels
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        num_imgs = len(batch_img_metas)
        featmap_sizes = [pred_map.shape[-2:] for pred_map in pred_maps]

        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=pred_maps[0].device)
        flatten_preds = []
        flatten_strides = []
        for pred, stride in zip(pred_maps, self.featmap_strides):
            pred = pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.num_attrib)
            pred[..., :2].sigmoid_()
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
        results_list = []
        for (bboxes, scores, objectness,
             img_meta) in zip(flatten_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):
            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            if conf_thr > 0:
                conf_inds = objectness >= conf_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            score_thr = cfg.get('score_thr', 0)
            nms_pre = cfg.get('nms_pre', -1)
            scores, labels, keep_idxs, _ = filter_scores_and_topk(
                scores, score_thr, nms_pre)

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
                score_factors=objectness[keep_idxs],
            )
            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms,
                img_meta=img_meta)
            results_list.append(results)
        return results_list

    def loss_by_feat(
            self,
            pred_maps: Sequence[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        device = pred_maps[0][0].device

        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]
        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [mlvl_anchors for _ in range(num_imgs)]

        responsible_flag_list = []
        for img_id in range(num_imgs):
            responsible_flag_list.append(
                self.responsible_flags(featmap_sizes,
                                       batch_gt_instances[img_id].bboxes,
                                       device))

        target_maps_list, neg_maps_list = self.get_targets(
            anchor_list, responsible_flag_list, batch_gt_instances)

        losses_cls, losses_conf, losses_xy, losses_wh = multi_apply(
            self.loss_by_feat_single, pred_maps, target_maps_list,
            neg_maps_list)

        return dict(
            loss_cls=losses_cls,
            loss_conf=losses_conf,
            loss_xy=losses_xy,
            loss_wh=losses_wh)

    def loss_by_feat_single(self, pred_map: Tensor, target_map: Tensor,
                            neg_map: Tensor) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

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

        loss_cls = self.loss_cls(pred_label, target_label, weight=pos_mask)
        loss_conf = self.loss_conf(
            pred_conf, target_conf, weight=pos_and_neg_mask)
        loss_xy = self.loss_xy(pred_xy, target_xy, weight=pos_mask)
        loss_wh = self.loss_wh(pred_wh, target_wh, weight=pos_mask)

        return loss_cls, loss_conf, loss_xy, loss_wh

    def get_targets(self, anchor_list: List[List[Tensor]],
                    responsible_flag_list: List[List[Tensor]],
                    batch_gt_instances: List[InstanceData]) -> tuple:
        """Compute target maps for anchors in multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_total_anchors, 4).
            responsible_flag_list (list[list[Tensor]]): Multi level responsible
                flags of each image. Each element is a tensor of shape
                (num_total_anchors, )
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - target_map_list (list[Tensor]): Target map of each level.
                - neg_map_list (list[Tensor]): Negative map of each level.
        """
        num_imgs = len(anchor_list)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        results = multi_apply(self._get_targets_single, anchor_list,
                              responsible_flag_list, batch_gt_instances)

        all_target_maps, all_neg_maps = results
        assert num_imgs == len(all_target_maps) == len(all_neg_maps)
        target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
        neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)

        return target_maps_list, neg_maps_list

    def _get_targets_single(self, anchors: List[Tensor],
                            responsible_flags: List[Tensor],
                            gt_instances: InstanceData) -> tuple:
        """Generate matching bounding box prior and converted GT.

        Args:
            anchors (List[Tensor]): Multi-level anchors of the image.
            responsible_flags (List[Tensor]): Multi-level responsible flags of
                anchors
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            tuple:
                target_map (Tensor): Predication target map of each
                    scale level, shape (num_total_anchors,
                    5+num_classes)
                neg_map (Tensor): Negative map of each scale level,
                    shape (num_total_anchors,)
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        anchor_strides = []
        for i in range(len(anchors)):
            anchor_strides.append(
                torch.tensor(self.featmap_strides[i],
                             device=gt_bboxes.device).repeat(len(anchors[i])))
        concat_anchors = torch.cat(anchors)
        concat_responsible_flags = torch.cat(responsible_flags)

        anchor_strides = torch.cat(anchor_strides)
        assert len(anchor_strides) == len(concat_anchors) == \
               len(concat_responsible_flags)
        pred_instances = InstanceData(
            priors=concat_anchors, responsible_flags=concat_responsible_flags)

        assign_result = self.assigner.assign(pred_instances, gt_instances)
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        target_map = concat_anchors.new_zeros(
            concat_anchors.size(0), self.num_attrib)

        target_map[sampling_result.pos_inds, :4] = self.bbox_coder.encode(
            sampling_result.pos_priors, sampling_result.pos_gt_bboxes,
            anchor_strides[sampling_result.pos_inds])

        target_map[sampling_result.pos_inds, 4] = 1

        gt_labels_one_hot = F.one_hot(
            gt_labels, num_classes=self.num_classes).float()
        if self.one_hot_smoother != 0:  # label smooth
            gt_labels_one_hot = gt_labels_one_hot * (
                1 - self.one_hot_smoother
            ) + self.one_hot_smoother / self.num_classes
        target_map[sampling_result.pos_inds, 5:] = gt_labels_one_hot[
            sampling_result.pos_assigned_gt_inds]

        neg_map = concat_anchors.new_zeros(
            concat_anchors.size(0), dtype=torch.uint8)
        neg_map[sampling_result.neg_inds] = 1

        return target_map, neg_map

    def responsible_flags(self, featmap_sizes: List[tuple], gt_bboxes: Tensor,
                          device: str) -> List[Tensor]:
        """Generate responsible anchor flags of grid cells in multiple scales.

        Args:
            featmap_sizes (List[tuple]): List of feature map sizes in multiple
                feature levels.
            gt_bboxes (Tensor): Ground truth boxes, shape (n, 4).
            device (str): Device where the anchors will be put on.

        Return:
            List[Tensor]: responsible flags of anchors in multiple level
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_responsible_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.prior_generator.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            gt_cx = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) * 0.5).to(device)
            gt_cy = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) * 0.5).to(device)
            gt_grid_x = torch.floor(gt_cx / anchor_stride[0]).long()
            gt_grid_y = torch.floor(gt_cy / anchor_stride[1]).long()
            # row major indexing
            gt_bboxes_grid_idx = gt_grid_y * feat_w + gt_grid_x

            responsible_grid = torch.zeros(
                feat_h * feat_w, dtype=torch.uint8, device=device)
            responsible_grid[gt_bboxes_grid_idx] = 1

            responsible_grid = responsible_grid[:, None].expand(
                responsible_grid.size(0),
                self.prior_generator.num_base_priors[i]).contiguous().view(-1)

            multi_level_responsible_flags.append(responsible_grid)
        return multi_level_responsible_flags
