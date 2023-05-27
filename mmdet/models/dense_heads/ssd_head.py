# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import force_fp32

from mmdet.core import (build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, multi_apply)
from ..builder import HEADS
from ..losses import smooth_l1_loss
from .anchor_head import AnchorHead


# TODO: add loss evaluator for SSD
@HEADS.register_module()
class SSDHead(AnchorHead):
    """SSD head used in https://arxiv.org/abs/1512.02325.

    Args:
        num_classes (int): 不包括背景类别的类别数.
        in_channels (int): 输入特征图中的通道数.
        stacked_convs (int): Number of conv layers in cls and reg tower.
        feat_channels (int): 当stacked_convs > 0 时的隐藏通道数.
        use_depthwise (bool): 是否使用 DepthwiseSeparableConv.
        conv_cfg (dict): 构造和配置conv层的字典.
        norm_cfg (dict): 构造和配置norm层的字典.
        act_cfg (dict): 构造和配置激活层的字典.
        anchor_generator (dict): anchor生成器的配置字典
        bbox_coder (dict): box编解码的配置.
        reg_decoded_bbox (bool): 为True时,则将box的绝对坐标与gt的绝对坐标作为loss计算.
            一般为IOU类loss时为True.
        train_cfg (dict): anchor head的训练阶段配置.
        test_cfg (dict): anchor head的测试阶段配置.
        init_cfg (dict or list[dict], optional): 初始化配置字典.
    """  # noqa: W605

    def __init__(self,
                 num_classes=80,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 stacked_convs=0,
                 feat_channels=256,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 anchor_generator=dict(
                     type='SSDAnchorGenerator',
                     scale_major=False,
                     input_size=300,
                     strides=[8, 16, 32, 64, 100, 300],
                     ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                     basesize_ratio_range=(0.1, 0.9)),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[.0, .0, .0, .0],
                     target_stds=[1.0, 1.0, 1.0, 1.0],
                 ),
                 reg_decoded_bbox=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Xavier',
                     layer='Conv2d',
                     distribution='uniform',
                     bias=0)):
        super(AnchorHead, self).__init__(init_cfg)  # 略过AnchorHead类的初始化
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.use_depthwise = use_depthwise
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.cls_out_channels = num_classes + 1  # 添加背景类
        self.prior_generator = build_prior_generator(anchor_generator)

        # 通常大部分dense head每个层级的anchor数量是相同的,比如[n,]*num_level,
        # 但是SSD则是[4, 6, 6, 6, 4, 4]代表各层级的基础anchor数量
        self.num_base_priors = self.prior_generator.num_base_priors

        self._init_layers()

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # 初始化采样配置
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # 当sampling=False时,默认使用PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

    def _init_layers(self):
        """初始化SSD head层权重."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # TODO: Use registry to choose ConvModule type
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule

        # 理论上应生成 6, 10, 10, 10, 6, 6个anchor.实际为4, 6, 6, 6, 4, 4
        # 根据六个特征图的输入通道数,以及每个特征图上特征点对应的anchor数量生成conv
        # 一般情况下(SSD300),每个特征图对应一个3x3的回归与分类卷积
        for channel, num_base_priors in zip(self.in_channels,
                                            self.num_base_priors):
            cls_layers = []
            reg_layers = []
            in_channel = channel
            # build stacked conv tower, not used in default ssd
            for i in range(self.stacked_convs):
                cls_layers.append(
                    conv(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_layers.append(
                    conv(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                in_channel = self.feat_channels
            # SSD-Lite head
            if self.use_depthwise:
                cls_layers.append(
                    ConvModule(
                        in_channel,
                        in_channel,
                        3,
                        padding=1,
                        groups=in_channel,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_layers.append(
                    ConvModule(
                        in_channel,
                        in_channel,
                        3,
                        padding=1,
                        groups=in_channel,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            cls_layers.append(
                nn.Conv2d(
                    in_channel,
                    num_base_priors * self.cls_out_channels,
                    kernel_size=1 if self.use_depthwise else 3,
                    padding=0 if self.use_depthwise else 1))
            reg_layers.append(
                nn.Conv2d(
                    in_channel,
                    num_base_priors * 4,
                    kernel_size=1 if self.use_depthwise else 3,
                    padding=0 if self.use_depthwise else 1))
            self.cls_convs.append(nn.Sequential(*cls_layers))
            self.reg_convs.append(nn.Sequential(*reg_layers))

    def forward(self, feats):
        """对来自上游的特征进行并行前向传播.SSD的Head权重在不同层级间独立.
            SSD300,各层级特征上基础anchor个数为[4, 6, 6, 6, 4, 4]
        Args:
            feats (tuple[Tensor]): 来自上游网络的多层级特征.
                SSD300:tuple(torch.Size([8, 512, 38, 38])
                            torch.Size([8, 1024, 19, 19])
                            torch.Size([8, 512, 10, 10])
                            torch.Size([8, 256, 5, 5])
                            torch.Size([8, 256, 3, 3])
                            torch.Size([8, 256, 1, 1]))
        Returns:
            tuple:
                cls_scores (list[Tensor]): 所有层级特征图的cls_score.
                    [[bs, na*(nc+1)), h, w], ] * num_level
                bbox_preds (list[Tensor]): 所有层级特征图的box_reg.
                    [[bs, na*4, h, w], ] * num_level
        """
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    def loss_single(self, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """计算单个图像的loss.

        Args:
            cls_score (Tensor): 单幅图像生成的box cls, [num_level*h*w*na, nc+1]
            bbox_pred (Tensor): 单幅图像生成的box reg, [num_level*h*w*na,4]
            anchor (Tensor): 单幅图像生成的anchor, [num_level*h*w*na,4]
            labels (Tensor): 单幅图像上anchor对应的gt label [num_level*h*w*na,]
            label_weights (Tensor): 每个anchor label的权重 [num_level*h*w*na,]
            bbox_targets (Tensor): 单幅图像拟合的box reg,[num_level*h*w*na,4]
            bbox_weights (Tensor): 每个box reg的权重 [num_level*h*w*na,4]
            num_total_samples (int): SSD计算loss时作为分母的样本数量固定为正样本数量.

        Returns:
            dict[str, Tensor]: 计算得到的loss字典.
        """

        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        # 前景类: [0, num_classes -1], 背景类: num_classes
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero().reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero().view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        # SSD在计算分类损失时,先计算出所有的负样本的loss.记 N=neg_pos_ratio * num_pos_samples
        # 然后挑选前N个作为负样本的loss与正样本的loss一起当做分类总loss
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        if self.reg_decoded_bbox:  # 为True时计算绝对坐标的loss,否则计算修正系数的loss
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        # 不太明白此处[None]的意义
        return loss_cls[None], loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """计算head部分的损失.

        Args:
            cls_scores (list[Tensor]): 所有层级的box score
                 [(bs, na * (nc+1), h, w),] * num_level
            bbox_preds (list[Tensor]): 所有层级的box reg
                [(bs, na * 4, h, w),] * num_level
            gt_bboxes (list[Tensor]): [[num_gt, 4], ] * bs, batch幅图像的gt box.
            gt_labels (list[Tensor]): [[num_gt, ], ] * bs, batch幅图像的gt label.
            img_metas (list[dict]): 每张图片的元信息. 以SSD300为例,参考如下
                {'filename': 'd:/mmdetection/data/yexi/images/001061.jpg',  修改后的图片绝对路径
                'ori_filename': '001061.jpg',   原始数据中的图片名称
                'ori_shape': (600, 800, 3),     原始图片的尺寸
                'img_shape': (300, 300, 3),     Resize后的图片尺寸
                'pad_shape': (300, 300, 3),     Pad后的图片尺寸
                'scale_factor': array([0.38659793, 0.75949365, 0.38659793, 0.75949365], dtype=float32),
                 resize时,box各个边对应的缩放比例
                'flip': False,                  图像是否翻转
                'flip_direction': None,         如果翻转,那么翻转方式是什么
                'img_norm_cfg': {               图像归一化的配置
                                'mean': array([123.675, 116.28 , 103.53 ], dtype=float32),
                                'std': array([1., 1., 1.], dtype=float32), 'to_rgb': True},
                'batch_input_shape': (300, 300)}网络模型前向传播时的batch shape
            gt_bboxes_ignore (None | list[Tensor]): 计算损失时可以指定忽略哪些gt box.

        Returns:
            dict[str, Tensor]: 计算出的各loss字典.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        # [[[h * w * na, 4], ] * num_levels,] * bs, [[[h * w * na, ], ] * num_levels, ] * bs
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=True)
        if cls_reg_targets is None:
            return None
        # labels_list -> [[bs, h * w * na, ], ] * num_level. 其他同理
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        # [bs,h*w*na,nc+1] * num_level -> [bs,num_level*h*w*na,nc+1]
        all_cls_scores = torch.cat([
            # [bs,na*(nc+1),h,w] -> [bs,h,w,na*(nc+1)] -> [bs,h*w*na,nc+1]
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        # [[bs, h * w * na], ] * num_level -> [bs, num_level * h * w * na]
        all_labels = torch.cat(labels_list, -1)
        all_label_weights = torch.cat(label_weights_list, -1)
        all_bbox_preds = torch.cat([  # 与 all_cls_scores 同理
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        # [[bs, h * w * na, 4], ] * num_level -> [bs, num_level * h * w * na, 4]
        all_bbox_targets = torch.cat(bbox_targets_list, -2)
        all_bbox_weights = torch.cat(bbox_weights_list, -2)

        all_anchors = []  # 最终为[[num_level*h*w*na, 4], ] * bs
        for i in range(num_images):  # SSD300,每张图片上的anchor的shape都是[8732, 4].
            all_anchors.append(torch.cat(anchor_list[i]))

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
