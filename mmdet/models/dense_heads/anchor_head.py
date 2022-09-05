# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, unmap)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class AnchorHead(BaseDenseHead, BBoxTestMixin):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): 不包括背景类的总类别数.
        in_channels (int): 输入特征图的通道数.
        feat_channels (int): 隐藏通道数,在子类中使用.
        anchor_generator (dict): anchor generator的配置字典.
        bbox_coder (dict): bounding box coder的配置字典.
        reg_decoded_bbox (bool): 为True时,网络拟合绝对坐标.
            一般回归损失是`IouLoss`,`GIouLoss`等IOU类损失
            为False时,网络拟合修正系数.一般回归损失是`L1Loss`,`Smooth L1Loss`等
        loss_cls (dict): 分类loss的配置字典.
        loss_bbox (dict): 回归loss的配置字典.
        train_cfg (dict): anchor head的训练配置字典.
        test_cfg (dict): anchor head的测试配置字典.
        init_cfg (dict or list[dict], optional): 初始化配置字典.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     strides=[4, 8, 16, 32, 64]),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 reg_decoded_bbox=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01)):
        super(AnchorHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg,
                       'sampler') and self.train_cfg.sampler.type.split(
                           '.')[-1] != 'PseudoSampler':
                self.sampling = True
                sampler_cfg = self.train_cfg.sampler
                # avoid BC-breaking
                if loss_cls['type'] in [
                        'FocalLoss', 'GHMC', 'QualityFocalLoss'
                ]:
                    warnings.warn(
                        'DeprecationWarning: Determining whether to sampling'
                        'by loss type is deprecated, please delete sampler in'
                        'your config when using `FocalLoss`, `GHMC`, '
                        '`QualityFocalLoss` or other FocalLoss variant.')
                    self.sampling = False
                    sampler_cfg = dict(type='PseudoSampler')
            else:
                self.sampling = False
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        self.prior_generator = build_prior_generator(anchor_generator)

        # 通常每个层级的基础anchor数量是相同的,除了SSD. 大部分模型的head的都是一个整数,
        # 而SSDHead的各个层级的基础anchor则是一个列表,每个层级基础anchor数量不同
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self._init_layers()

    @property
    def num_anchors(self):
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, '
                      'for consistency or also use '
                      '`num_base_priors` instead')
        return self.prior_generator.num_base_priors[0]

    @property
    def anchor_generator(self):
        warnings.warn('DeprecationWarning: anchor_generator is deprecated, '
                      'please use "prior_generator" instead')
        return self.prior_generator

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_cls = nn.Conv2d(self.in_channels,
                                  self.num_base_priors * self.cls_out_channels,
                                  1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_base_priors * 4,
                                  1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        """对来自上游网络的特征图进行前向传播.

        Args:
            feats (tuple[Tensor]): 来自上游网络的特征图,每个都是 4维 张量.

        Returns:
            tuple: tuple(cls_scores, bbox_preds).

                - cls_scores (list[Tensor]): 所有层级的cls_scores,每个都是4D张量,
                    通道数是num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): 所有层级的box_predict,每个都是4D张量,
                    通道数是num_base_priors * 4.
        """
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """根据特征图尺寸获取anchors(多层级).

        Args:
            featmap_sizes (list[tuple]): 多层级特征图尺寸.
            img_metas (list[dict]): 图像元信息.
            device (torch.device | str): 返回张量所在的设备

        Returns:
            tuple:
                anchor_list (list[Tensor]): 单张图像的anchor.
                valid_flag_list (list[Tensor]): 每张图像的有效mask.
        """
        num_imgs = len(img_metas)

        # 由于一个batch中所有图像的特征图大小相同,我们这里只计算一次anchor,然后复制即可
        # [(feat_h*feat_w*A,4)]*self.num_levels,注意anchor数量和尺寸随着level变化而变化
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # 对于每副图像,我们需要计算出多个特征图上的anchor的有效索引
        valid_flag_list = []  # [[(f_h*f_w*A),]*self.num_levels]*num_imgs
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """计算单张图像上anchor的回归和分类拟合目标.

        Args:
            flat_anchors (Tensor): 被合并到一起的单张图像的多层级anchor, (num_levels*feat_h*feat_w*A ,4)
            valid_flags (Tensor): 同上,有效anchor的mask, (num_levels*feat_h*feat_w*A ,)
            gt_bboxes (Tensor): 单张图片的真实框, (num_gts, 4)
            gt_bboxes_ignore (Tensor): 单张图片中将要被忽略的gt box, (num_ignored_gts, 4)
            img_meta (dict): 单张图片的元信息
            gt_labels (Tensor): 单张图片中真实框的所属类别, (num_gts,)
            label_channels (int): 类别总数
            unmap_outputs (bool): 是否将计算出的有效anchor的label、target、weight映射回原始anchor上

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        # flat_anchors -> (feat_h*feat_w*A*self.num_levels,4)
        # 注意feat_h*feat_w随着level变化而变化,valid_flags过滤了
        # 在collact阶段填充的多余像素区域生成的anchor,而这一步是过滤掉超出图像边界的一些anchor
        # self.train_cfg.allowed_border默认为-1,即不进行过滤,
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        # 先对单张图片上的有效anchor进行分配正负样本及背景
        # 若采用sampling,那么不需要传递gt_labels.因为后面还会再次计算assigned_labels
        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        # 然后进行采样以控制正负样本比例
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),  # 初始化为self.num_classes(背景类别)
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # 从 v2.5.0开始,只有 rpn 会将 gt_labels 设为 None,
                # 此时是一个二分类网络,前景是第一个类别
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

        # 是否将图像内部的anchor映射回原始anchor
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # 默认值为背景类别
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """计算多张图像中anchor的回归和分类目标.

        Args:
            anchor_list (list[list[Tensor]]): [[(feat_h*feat_w*A,4)]*self.num_levels]*B.
            valid_flag_list (list[list[Tensor]]):[[(f_h*f_w*A,)]*self.num_levels]*B
            gt_bboxes_list (list[Tensor]): 每张图像中的标注box.
            img_metas (list[dict]): 每张图片的元信息.
            gt_bboxes_ignore_list (list[Tensor]): batch幅图像的要忽略的gt box列表.
                其内部元素shape为 (num_ignored_gts, 4).意为该幅图片所忽略的gt box
            gt_labels_list (list[Tensor]): 每张图像中标注box所属类别.
            label_channels (int): 类别维度.
            unmap_outputs (bool): 是否将输出映射回原始anchor上.

        Returns:
            tuple: 通常返回一个包含网络拟合目标的元组.

                - labels_list (list[Tensor]): 每个层级上对应的label.
                    [B,feat_h*feat_w*A] * num_level feat_h*feat_w随着层级不同而不同
                - label_weights_list (list[Tensor]): 每个层级上的label权重.
                    [B,feat_h*feat_w*A] * num_level
                - bbox_targets_list (list[Tensor]): 每个层级上特征图需要回归的目标.
                    [B,feat_h*feat_w*A, 4] * num_level
                - bbox_weights_list (list[Tensor]): 每个层级上特征图回归的权重.
                    [B,feat_h*feat_w*A, 4] * num_level
                - num_total_pos (int): batch幅图像中的正样本数.
                - num_total_neg (int): batch幅图像中的负样本数.

            额外返回值: 此值来自 `self._get_targets_single` 的用户定义返回.
                这些返回值目前被细化为每个特征图的属性(即具有 HxW 维度).它将与原始返回值
                合并到一起返回.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        # 单张图片的各个层级上的anchor总数量,(向上取整),以retinanet-fpn为例
        # [h/8*w/8*9, h/16*w/16*9, h/32*w/32*9, h/64*w/64*9, h/128*w/128*9]
        # [43200, 10800, 2700, 720, 180],其中网络输入batch尺寸(h,w)=(480, 640)
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # 将一张图片上的所有anchor全部cat到一起
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # 计算每张图像的拟合目标
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,  # 在单张图片上计算anchor目标
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # 没有有效的anchor ,出现该种情况的原因:
        # 直接原因:
        #   在前向传播时,batch中某一幅图像边界内没有任何"有效"anchor.
        # 根本原因：
        #   训练数据的原始图像尺寸比例过于悬殊,比如1000x150.此时生成的anchor很容易过界
        #   anchor、allowed_border等参数不合理
        # 你需要检查图像比例和anchor大小,也可以将配置文件中的allowed_border设置为-1,以跳过筛查条件
        # 参考:https://github.com/open-mmlab/mmdetection/issues/1882#issuecomment-569276188
        # 注意!遇到该种情况,虽然可能仅仅由于一张图片引发的,但整体情况已经很严重了.需要重新审视数据或参数
        if any([labels is None for labels in all_labels]):
            return None
        # 整个batch图像上经过sample的正负anchor
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # 将targets拆分为一个列表 与各个特征图层级对应
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """计算单个层级的损失.其中B代表batch_size,A代表每个特征点上的基础anchor数量
            H代表该层级特征图的高度,W为宽度

        Args:
            cls_score (Tensor): 每层级输出的类别置信度 (B, A * num_cls, H, W).
            bbox_pred (Tensor): 每层级输出的box/loc  (B, A * 4, H, W).
            anchors (Tensor): 每层级的基础anchor     (B, H * W * A, 4).
            labels (Tensor): 每层级的类别拟合目标     (B, H * W * A,).
            label_weights (Tensor): 每层级的类别权重 (B, H * W * A,).
            bbox_targets (Tensor): 每层级的回归拟合目标  (B, H * W * A, 4).
            bbox_weights (Tensor): 每层级的回归权重  (B, H * W * A, 4).
            num_total_samples (int): 如果经过采样,则该值等于正负样本总数;
                否则,它是正样本的数量.

        Returns:
            dict[str, Tensor]: 分类和回归损失.
        """
        # 分类损失
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # 回归损失
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        # 获取batch图像上的所有anchor,以及有效anchor的mask(在collate阶段填充的区域被视为无效区域)
        # [[(feat_h*feat_w*A,4)]*self.num_levels]*B, [[(f_h*f_w*A)]*self.num_levels]*B.
        # 注意以上两个变量,前者只要计算一张图像上的anchor然后复制B次,因为同一batch下所有图像尺寸一致
        # 而后者依据每幅图片的尺寸不同,anchor有效区域也可能不同,需要具体计算出每张图像的有效mask.
        # 以及feat_h*feat_w是随着层级(level)变化而变化的
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        # 利用提供的anchor将其与每张图片上的target计算IOU以及利用相应的thr来分配正负样本及背景
        # 接着采取sample采样正负样本,主要用来筛选合适的正负样本用来训练.同时为他们分配最佳gt和label
        # 最后将数据进行调整以和网络前向传播结果保持一致,方便loss计算
        cls_reg_targets = self.get_targets(  # [level0,level1,...]
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # 如果样本经过手动采样那么会控制负样本数量,否则正负样本比例可能悬殊会导致loss不好收敛
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # 各层级anchor数量列表
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # 现将每个图像上的anchor合并到一起,然后由img级别映射到level级别,该操作在get_targets
        # 方法最后部分也存在.之所以这么做的原因是为了和网络输出shape保持一致,方便计算loss
        concat_anchor_list = []
        for i in range(len(anchor_list)):  # 现将每个图像上的anchor合并到一起
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5), where
                5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,), The length of list should always be 1.
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
