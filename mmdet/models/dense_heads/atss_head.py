# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList, reduce_mean)
from ..task_modules.prior_generators import anchor_inside_flags
from ..utils import images_to_levels, multi_apply, unmap
from .anchor_head import AnchorHead


@MODELS.register_module()
class ATSSHead(AnchorHead):
    """Detection Head of `ATSS <https://arxiv.org/abs/1912.02424>`_.

    需要注意的一点是,MMDetection中的ATSS是基于 RetinaNet 为例的.它的Head与FCOS相似,
    但它是anchor-based的,并通过"自适应训练样本选择"分配正负样本,这点又与RetinaNet不同.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        pred_kernel_size (int): Kernel size of ``nn.Conv2d``
        stacked_convs (int): Number of stacking convs of the head.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to ``dict(type='GN', num_groups=32,
            requires_grad=True)``.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Defaults to False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_centerness (:obj:`ConfigDict` or dict): Config of centerness loss.
            Defaults to ``dict(type='CrossEntropyLoss', use_sigmoid=True,
            loss_weight=1.0)``.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 pred_kernel_size: int = 3,
                 stacked_convs: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 reg_decoded_bbox: bool = True,
                 loss_centerness: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='atss_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.pred_kernel_size = pred_kernel_size
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            reg_decoded_bbox=reg_decoded_bbox,
            init_cfg=init_cfg,
            **kwargs)

        self.sampling = False
        self.loss_centerness = MODELS.build(loss_centerness)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        pred_pad_size = self.pred_kernel_size // 2
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.atss_reg = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 4,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.atss_centerness = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 1,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): [[bs, c, h, w], ] * nl

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): cls_score,
                    [bs,num_prior_lvl * nc, h, w],] * nl
                bbox_preds (list[Tensor]): box_reg,
                    [bs,num_prior_lvl * 4, h, w],] * nl
        """
        return multi_apply(self.forward_single, x, self.scales)

    def forward_single(self, x: Tensor, scale: Scale) -> Sequence[Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): 用于调整reg大小的可学习参数.

        Returns:
            tuple:
                cls_score (Tensor): [bs, na * nc, h, w].
                bbox_pred (Tensor): [bs, na * 4, h, w].
                centerness (Tensor): [bs, na * 1, h, w].
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # 我们只关心ATSS的实现,不会在reg上应用FCOS中的exp或者stride
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness

    def loss_by_feat_single(self, anchors: Tensor, cls_score: Tensor,
                            bbox_pred: Tensor, centerness: Tensor,
                            labels: Tensor, label_weights: Tensor,
                            bbox_targets: Tensor, avg_factor: float) -> dict:
        """计算单层级上的损失.

        Args:
            cls_score (Tensor): anchor cls, [bs, na * nc, h, w].
            bbox_pred (Tensor): anchor reg, [bs, na * 4, h, w].
            anchors (Tensor): [bs, h * w * na, 4].
            labels (Tensor): target cls, [bs, h * w * na].
            label_weights (Tensor): anchor cls计算loss时的权重, [bs, h * w * na].
            bbox_targets (Tensor): target box, [bs, h * w * na, 4].
            avg_factor (float): 平均因子.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # cls loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)

        # 前景: [0, nc), 背景: nc
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            # 因为self.centerness_target方法中形参gt格式为绝对坐标,
            # 所以这里代码被写死了,固定为iou类loss, 即回归绝对坐标
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)

            # reg loss,注意这里avg_factor固定为1,涉及到分布式训练,它后面先是计算出
            # centerness在所有层级所有batch幅图像上的总和,再将各层级上的reg loss除以该值
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)

            # 它本质上是个obj loss
            loss_centerness = self.loss_centerness(
                pos_centerness, centerness_targets, avg_factor=avg_factor)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            centernesses: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): anchor cls, [[bs, na * nc, h, w],] * nl
            bbox_preds (list[Tensor]): anchor reg, [[bs, na * 4, h, w],] * nl
            centernesses (list[Tensor]): anchor obj, [[bs, na * 1, h, w],] * nl
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
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        # [[[[h * w * na, 4], ] * num_level], ] * bs  [[[[h*w*na, ], ] * num_level], ]*bs
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        # [[bs,h * w * na, 4], ] * nl
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets
        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, loss_centerness, \
            bbox_avg_factor = multi_apply(
                self.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                centernesses,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                avg_factor=avg_factor)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_centerness=loss_centerness)

    def centerness_target(self, anchors: Tensor, gts: Tensor) -> Tensor:
        """只能计算正样本的centerness targets, 因为正样本在gt box 内部
        而一旦超出gt box边界那么距离就会为负数,而torch.sqrt(负数)为nan

        Args:
            anchors (Tensor): Anchors with shape (N, 4), "xyxy" format.
            gts (Tensor): Ground truth bboxes with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Centerness between anchors and gts.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def get_targets(self,
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True) -> tuple:
        """计算head的cls_target, reg_target.

        anchor_list: [[[[h * w * na, 4], ] * nl], ] * bs
        valid_flag_list: [[[[h * w * na, ], ] * nl,] ] * bs

        该方法与`AnchorHead.get_targets()`几乎相同. 除了像父方法一样返回目标,
        它还将anchor作为返回元组的第一个元素返回.
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # 各层级上anchor数量
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # anchor_list -> [[nl*h*w*na, 4], ]*bs valid_flag_list同理
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # 计算每幅图像的拟合目标
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list,
         sampling_results_list) = multi_apply(
             self._get_targets_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             batch_gt_instances,
             batch_img_metas,
             batch_gt_instances_ignore,
             unmap_outputs=unmap_outputs)
        # Get `avg_factor` of all images, which calculate in `SamplingResult`.
        # When using sampling method, avg_factor is usually the sum of
        # positive and negative priors. When using `PseudoSampler`,
        # `avg_factor` is usually equal to the number of positive priors.
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        # [[nl*h*w*na, 4], ] * bs -> [[bs,h * w * na, 4], ] * nl
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, avg_factor)

    def _get_targets_single(self,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            num_level_anchors: List[int],
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """计算单幅图像上的reg_target与cls_target.

        Args:
            flat_anchors (Tensor): [nl*h*w*na, 4]
            valid_flags (Tensor): flat_anchors对应的有效flag,[nl*h*w*na, ].
            num_level_anchors (List[int]): [h*w*na, ] * nl.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): 是否将输出映射回原始anchor.

        Returns:
            tuple:
                labels (Tensor): target cls, [nl*h*w*na, ].
                label_weights (Tensor): anchor cls在计算loss时的权重,[nl*h*w*na, ].
                bbox_targets (Tensor): target box/reg,[num_level*h*w*na, 4].
                bbox_weights (Tensor): target box/reg在计算loss时的权重,[nl*h*w*na, 4].
                pos_inds (Tensor): 正样本的anchor索引 [num_pos, ].
                neg_inds (Tensor): 负样本的anchor索引 [num_pos, ].
                sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        # 通过利用torch.split方法来获取每个层级上合格的anchor数量
        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances,
                                             num_level_anchors_inside,
                                             gt_instances, gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        # 先初始化cls_label皆为背景
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox:  # 为True代表回归绝对坐标,否则回归修正系数
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # 映射回原始anchor上
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, sampling_result)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        """Get the number of valid anchors in every level."""

        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
