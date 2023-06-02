# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead


@HEADS.register_module()
class ATSSHead(AnchorHead):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    需要注意的一点是,MMDetection中的ATSS是基于 RetinaNet 为例的.它的Head与FCOS相似,
    但它是anchor-based的,并通过"自适应训练样本选择"分配正负样本,这点又与RetinaNet不同.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 pred_kernel_size=3,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 reg_decoded_bbox=True,
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='atss_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.pred_kernel_size = pred_kernel_size
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(ATSSHead, self).__init__(
            num_classes,
            in_channels,
            reg_decoded_bbox=reg_decoded_bbox,
            init_cfg=init_cfg,
            **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler 不太明白这句注释
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self):
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
            self.num_base_priors * self.cls_out_channels,
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

    def forward(self, feats):
        """对来自上游网络的特征图进行前向传播.

        Args:
            feats (tuple[Tensor]): ([bs, 256, h, w], ) * num_level
                注意,每次下采样如果输入为奇数则输出向上取整.h/w随着level不同而不同

        Returns:
            tuple: tuple(cls_scores, reg_predict).
                cls_scores (list[Tensor]): 所有层级的cls_scores,
                    [bs,num_priors_lvl * num_classes,h,w],] * num_level
                reg_predict (list[Tensor]): 所有层级的box_reg,
                    [bs,num_priors_lvl * 4,h,w],] * num_level
        """
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """单层级上的前向传播.

        Args:
            x (Tensor): 指定stride的 FPN 特征图.
            scale (:obj: `mmcv.cnn.Scale`): 用于调整回归大小的可学习比例模块.

        Returns:
            tuple:
                cls_score (Tensor): [N, num_anchors * num_classes, H, W].
                bbox_pred (Tensor): [N, num_anchors * 4, H, W].
                centerness (Tensor): [N, num_anchors * 1, H, W].
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

    def loss_single(self, anchors, cls_score, bbox_pred, centerness, labels,
                    label_weights, bbox_targets, num_total_samples):
        """计算单层级上的损失.

        Args:
            anchors (Tensor): 单层级上的anchor [bs, h * w * na, 4].
            cls_score (Tensor): 单层级上box cls [bs, na * nc, h, w].
            bbox_pred (Tensor): 单层级上box reg [bs, na * 4, h, w].
            centerness (Tensor): 单层级上的centerness [bs, na, h, w].
            labels (Tensor): 单层级上的cls_target [bs, h * w * na].
            label_weights (Tensor): 单层级上的cls_weight [bs, h * w * na].
            bbox_targets (Tensor): 单层级上的reg_target [bs, h * w * na, 4].
            num_total_samples (int): 在所有GPU上的平均正样本数量.

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
            cls_score, labels, label_weights, avg_factor=num_total_samples)

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
                pos_centerness,
                centerness_targets,
                avg_factor=num_total_samples)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """

        Args:
            cls_scores (list[Tensor]): 所有层级的cls_scores,
                [[bs, na * nc, h, w],] * num_level
            bbox_preds (list[Tensor]): 所有层级的box_reg,
                [[bs, na * 4, h, w],] * num_level
            centernesses (list[Tensor]): 所有层级的centernesses,
                [[bs, na, h, w],] * num_level
            gt_bboxes (list[Tensor]): batch张图像的gt box,格式为[x1, y1, x2, y2].
                [(num_gts, 4), ] * bs.注num_gts取决于每张图片.而非定值
            gt_labels (list[Tensor]): batch张图像上gt box对应的class.
                [(num_gts,), ] * bs.同上
            img_metas (list[dict]): batch张图片的元信息.
            gt_bboxes_ignore (None | list[Tensor]): [[num_ignore, 4], ] * bs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        # [[[[h * w * na, 4], ] * num_level], ] * bs  [[[[h*w*na, ], ] * num_level], ]*bs
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        # 该变量在后续没有使用到,暂时不明白它的意义
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        # [[bs,h * w * na, 4], ] * num_level
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, loss_centerness,\
            bbox_avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                centernesses,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                num_total_samples=num_total_samples)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_centerness=loss_centerness)

    def centerness_target(self, anchors, gts):
        # 只能计算正样本的centerness targets, 因为正样本在gt box 内部
        # 而一旦超出gt box边界那么距离就会为负数,而torch.sqrt(负数)为nan
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
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """计算head的cls_target, reg_target.

        anchor_list: [[[[h * w * na, 4], ] * num_level], ] * bs
        valid_flag_list: [[[[h * w * na, ], ] * num_level,] ] * bs

        该方法与`AnchorHead.get_targets()`几乎相同. 除了像父方法一样返回目标,
        它还将anchor作为返回元组的第一个元素返回.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # 各层级上anchor数量
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # anchor_list -> [[num_level*h*w*na, 4], ]*bs valid_flag_list同理
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # 计算每幅图像的拟合目标
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # 没有有效anchor
        if any([labels is None for labels in all_labels]):
            return None
        # batch幅图像中经过采样的正负样本数
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # 将一系列值由[[num_level*h*w*na, 4], ] * bs转换为[[bs,h * w * na, 4], ] * num_level
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """计算单幅图像上的reg_target与cls_target.

        Args:
            flat_anchors (Tensor): [num_level*h*w*na, 4]
            valid_flags (Tensor): [num_level*h*w*na, ]. 在非Padding区域的anchor mask
            num_level_anchors (list[int]): [h*w*na, ] * num_level.
            gt_bboxes (Tensor): [num_gts, 4].
            gt_bboxes_ignore (Tensor): [num_ignored_gts, 4].
            gt_labels (Tensor): [num_gts, ].
            img_meta (dict): 图像元信息.
            label_channels (int): 标签的类别数.
            unmap_outputs (bool): 是否将输出映射回原始anchor.

        Returns:
            tuple:
                anchors (Tensor): 所有层级的anchor [num_level*h*w*na, 4].
                labels (Tensor): anchor的cls_target [num_level*h*w*na, ].
                label_weights (Tensor): anchor的cls_weight [num_level*h*w*na, ].
                bbox_targets (Tensor): anchor的reg_target [num_level*h*w*na, 4].
                bbox_weights (Tensor): anchor的reg_weight [num_level*h*w*na, 4].
                pos_inds (Tensor): 正样本的anchor索引 [num_pos, ].
                neg_inds (Tensor): 负样本的anchor索引 [num_neg, ].
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        # 此时代表anchor设定完全不合理
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        # 通过利用torch.split方法来获取每个层级上合格的anchor数量
        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

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
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # 只有在RPN阶段gt_labels才为None,此时0为前景类别(v2.5.0开始)
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
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
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
