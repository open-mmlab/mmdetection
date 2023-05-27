# Copyright (c) OpenMMLab. All rights reserved.
from logging import warning
from math import ceil, log

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob
from mmcv.ops import CornerPool, batched_nms
from mmcv.runner import BaseModule, force_fp32

from mmdet.core import multi_apply
from ..builder import HEADS, build_loss
from ..utils import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import (gather_feat, get_local_maximum,
                                     get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


class BiCornerPool(BaseModule):
    """双向Corner Pooling模块(TopLeft, BottomRight, etc.)

    Args:
        in_channels (int): 模块输入通道数.
        out_channels (int): 模块输出通道数.
        feat_channels (int): 模块特征图通道数.
        directions (list[str]): 两个 CornerPool 的方向.
        norm_cfg (dict): 构造和配置norm层的字典.
        init_cfg (dict or list[dict], optional): 初始化配置字典.默认: None
    """

    def __init__(self,
                 in_channels,
                 directions,
                 feat_channels=128,
                 out_channels=128,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 init_cfg=None):
        super(BiCornerPool, self).__init__(init_cfg)
        self.direction1_conv = ConvModule(
            in_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        self.direction2_conv = ConvModule(
            in_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)

        self.aftpool_conv = ConvModule(
            feat_channels,
            out_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.conv1 = ConvModule(
            in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.conv2 = ConvModule(
            in_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg)

        self.direction1_pool = CornerPool(directions[0])
        self.direction2_pool = CornerPool(directions[1])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """CornerPool的前向传播函数.以下CBL为conv+bn+relu, CB为conv+bn
        ((x -> 3x3CBL -> top-cornerpool) + (x -> 3x3CBL -> left-cornerpool)
        -> 3x3CB) + (x -> 1x1CB) -> relu -> 3x3CBL
        direction1_pool其实就是对x指定维度翻转然后使用cummax函数而已,其中left/top为
        从左上角向右/下看(需要对x翻转),而right/bottom为从右下角向左/上看(不需要对x翻转).
        需要注意的是,进行pool完后要记得对x翻转回来(如果需要的话)
        Args:
            x (tensor): 输入特征图.

        Returns:
            conv2 (tensor): 输出特征图.
        """
        direction1_conv = self.direction1_conv(x)
        direction2_conv = self.direction2_conv(x)
        direction1_feat = self.direction1_pool(direction1_conv)
        direction2_feat = self.direction2_pool(direction2_conv)
        aftpool_conv = self.aftpool_conv(direction1_feat + direction2_feat)
        conv1 = self.conv1(x)
        relu = self.relu(aftpool_conv + conv1)
        conv2 = self.conv2(relu)
        return conv2


@HEADS.register_module()
class CornerHead(BaseDenseHead, BBoxTestMixin):
    """CornerNet的Head部分.

    代码修改自官方代码,<https://github.com/princeton-vl/CornerNet/blob/master/
    models/py_utils/kp.py#L73>.

    详情参考 <https://arxiv.org/abs/1808.01244>.

    Args:
        num_classes (int): 检测类别数.
        in_channels (int): 输入特征图中的通道数.
        num_feat_levels (int): 上一个模块的特征层级数. HourglassNet-104为2,
            HourglassNet-52为1. 因为HourglassNet-104 输出最终和中间特征图,
            而 HourglassNet-52 只输出最终特征图.
        corner_emb_channels (int): embedding vector的维度. 默认: 1.
        train_cfg (dict | None): 训练配置. 在CornerHead中没有使用,
            此参数的存在仅仅是为了兼容SingleStageDetector.
        test_cfg (dict | None): CornerHead 的测试配置. 默认: None.
        loss_heatmap (dict | None): heatmap loss的配置. 默认:GaussianFocalLoss.
        loss_embedding (dict | None): embedding loss的配置. 默认:
            AssociativeEmbeddingLoss.
        loss_offset (dict | None): offset loss的配置. 默认:SmoothL1Loss.
        init_cfg (dict or list[dict], optional): 初始化配置字典.默认: None
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_feat_levels=2,
                 corner_emb_channels=1,
                 train_cfg=None,
                 test_cfg=None,
                 loss_heatmap=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=1),
                 loss_embedding=dict(
                     type='AssociativeEmbeddingLoss',
                     pull_weight=0.25,
                     push_weight=0.25),
                 loss_offset=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1),
                 init_cfg=None):
        assert init_cfg is None, '为防止异常初始化行为,不允许设置init_cfg'
        super(CornerHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.corner_emb_channels = corner_emb_channels
        self.with_corner_emb = self.corner_emb_channels > 0
        self.corner_offset_channels = 2
        self.num_feat_levels = num_feat_levels
        self.loss_heatmap = build_loss(
            loss_heatmap) if loss_heatmap is not None else None
        self.loss_embedding = build_loss(
            loss_embedding) if loss_embedding is not None else None
        self.loss_offset = build_loss(
            loss_offset) if loss_offset is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fp16_enabled = False
        self._init_layers()

    def _make_layers(self, out_channels, in_channels=256, feat_channels=256):
        """为 CornerHead 初始化 conv 单元."""
        # CL3x3 + Conv1x1
        return nn.Sequential(
            ConvModule(in_channels, feat_channels, 3, padding=1),
            ConvModule(
                feat_channels, out_channels, 1, norm_cfg=None, act_cfg=None))

    def _init_corner_kpt_layers(self):
        """初始化keypoint层.

        包括heatmap和offset分支. 每个分支有两个部分: 左上角和右下角.
        """
        self.tl_pool, self.br_pool = nn.ModuleList(), nn.ModuleList()
        self.tl_heat, self.br_heat = nn.ModuleList(), nn.ModuleList()
        self.tl_off, self.br_off = nn.ModuleList(), nn.ModuleList()

        for _ in range(self.num_feat_levels):
            self.tl_pool.append(
                BiCornerPool(
                    self.in_channels, ['top', 'left'],
                    out_channels=self.in_channels))
            self.br_pool.append(
                BiCornerPool(
                    self.in_channels, ['bottom', 'right'],
                    out_channels=self.in_channels))

            self.tl_heat.append(
                self._make_layers(
                    out_channels=self.num_classes,
                    in_channels=self.in_channels))
            self.br_heat.append(
                self._make_layers(
                    out_channels=self.num_classes,
                    in_channels=self.in_channels))

            self.tl_off.append(
                self._make_layers(
                    out_channels=self.corner_offset_channels,
                    in_channels=self.in_channels))
            self.br_off.append(
                self._make_layers(
                    out_channels=self.corner_offset_channels,
                    in_channels=self.in_channels))

    def _init_corner_emb_layers(self):
        """初始化embedding层.

        仅包含左上和右下的embedding分支.
        """
        self.tl_emb, self.br_emb = nn.ModuleList(), nn.ModuleList()

        for _ in range(self.num_feat_levels):
            self.tl_emb.append(
                self._make_layers(
                    out_channels=self.corner_emb_channels,
                    in_channels=self.in_channels))
            self.br_emb.append(
                self._make_layers(
                    out_channels=self.corner_emb_channels,
                    in_channels=self.in_channels))

    def _init_layers(self):
        """初始化CornerHead.注意,head的权重在不同的层级间并不共享.

        包括两部分:keypoint层和embedding层
        """
        self._init_corner_kpt_layers()
        if self.with_corner_emb:
            self._init_corner_emb_layers()

    def init_weights(self):
        super(CornerHead, self).init_weights()
        bias_init = bias_init_with_prob(0.1)
        for i in range(self.num_feat_levels):
            # The initialization of parameters are different between
            # nn.Conv2d and ConvModule. Our experiments show that
            # using the original initialization of nn.Conv2d increases
            # the final mAP by about 0.2%
            self.tl_heat[i][-1].conv.reset_parameters()
            self.tl_heat[i][-1].conv.bias.data.fill_(bias_init)
            self.br_heat[i][-1].conv.reset_parameters()
            self.br_heat[i][-1].conv.bias.data.fill_(bias_init)
            self.tl_off[i][-1].conv.reset_parameters()
            self.br_off[i][-1].conv.reset_parameters()
            if self.with_corner_emb:
                self.tl_emb[i][-1].conv.reset_parameters()
                self.br_emb[i][-1].conv.reset_parameters()

    def forward(self, feats):
        """
        Args:
            feats (tuple[Tensor]): 来自HourglassNet-104(默认)的特征,注意CornerNet
                没有Neck,同时它是list而非tuple格式的数据,[[bs, 256, h, w],] * num_stack.

        Returns:
            tuple: 通常是一个heatmap元组,(corner, offset, embedding).
                - tl_heats (list[Tensor]): 所有层级的左上角热图,
                    [[bs, num_class, h, w],] * num_stack.
                - br_heats (list[Tensor]): 所有层级的右下角热图,
                    [[bs, num_class, h, w],] * num_stack.
                - tl_embs (list[Tensor] | list[None]): 所有层级的左上角embedding,
                  [[bs, corner_emb_channels, h, w],] * num_stack.
                - br_embs (list[Tensor] | list[None]): 所有层级的右下角embedding,
                  [[bs, corner_emb_channels, h, w],] * num_stack.
                - tl_offs (list[Tensor]): 所有层级的左上角embedding,
                  [[bs, corner_offset_channels, h, w],] * num_stack.
                - br_offs (list[Tensor]): 所有层级的右下角embedding,
                  [[bs, corner_offset_channels, h, w],] * num_stack.
        """
        lvl_ind = list(range(self.num_feat_levels))
        return multi_apply(self.forward_single, feats, lvl_ind)

    def forward_single(self, x, lvl_ind, return_pool=False):
        """单层级的前向传播.

        x -> corner pool -> heat/em/off 最后得到左上/右下各三个分支的输出张量
        Args:
            x (Tensor): 单层级特征. [bs, in_channel, h, w]
            lvl_ind (int): 当前特征的层级索引.
            return_pool (bool): 是否返回corner pool后的特征图.

        Returns:
            tuple[Tensor]: 当前层级输出的CornerHead元组. 包含以下张量:

                - tl_heat (Tensor): 预测的左上corner heatmap.[bs, num_class, h, w]
                - br_heat (Tensor): 预测的右下corner heatmap.[bs, num_class, h, w]
                - tl_emb (Tensor | None): 预测的左上corner embedding.
                    [bs, corner_emb_channels, h, w],`self.with_corner_emb == False`时为None.
                - br_emb (Tensor | None): 预测的右下corner embedding.
                    [bs, corner_emb_channels, h, w],`self.with_corner_emb == False`时为None.·
                - tl_off (Tensor): 预测的左上corner offset.[bs, 2, h, w]
                - br_off (Tensor): 预测的右下corner offset.[bs, 2, h, w]
                - tl_pool (Tensor): 左上corner pool后的特征图. 不一定存在.
                    [bs, in_channel, h, w], 注意corner pool操作不改变输入的维度.
                - br_pool (Tensor): 右下corner pool后的特征图. 不一定存在.
                    [bs, in_channel, h, w]
        """
        tl_pool = self.tl_pool[lvl_ind](x)
        tl_heat = self.tl_heat[lvl_ind](tl_pool)
        br_pool = self.br_pool[lvl_ind](x)
        br_heat = self.br_heat[lvl_ind](br_pool)

        tl_emb, br_emb = None, None
        if self.with_corner_emb:
            tl_emb = self.tl_emb[lvl_ind](tl_pool)
            br_emb = self.br_emb[lvl_ind](br_pool)

        tl_off = self.tl_off[lvl_ind](tl_pool)
        br_off = self.br_off[lvl_ind](br_pool)

        result_list = [tl_heat, br_heat, tl_emb, br_emb, tl_off, br_off]
        if return_pool:
            result_list.append(tl_pool)
            result_list.append(br_pool)

        return result_list

    def get_targets(self,
                    gt_bboxes,
                    gt_labels,
                    feat_shape,
                    img_shape,
                    with_corner_emb=False,
                    with_guiding_shift=False,
                    with_centripetal_shift=False):
        """生成corner target.

        包括 corner heatmap, corner offset.

        可选: corner embedding, corner guiding shift, centripetal shift.

        对于 CornerNet, 该函数额外生成corner embedding.

        对于 CentripetalNet, 该函数额外生成corner guiding shift 和 centripetal shift.

        Args:
            gt_bboxes (list[Tensor]): [[num_gt,4],] * bs.
            gt_labels (list[Tensor]): [[num_gt,],] * bs.
            feat_shape (list[int]): 网络输出特征的形状, [bs, _, h, w].
            img_shape (list[int]): pipline中Pad后的shape,[h, w, c].
            with_corner_emb (bool): 是否生成 corner embedding target.默认: False.
            with_guiding_shift (bool): 是否生成 guiding shift target.默认: False.
            with_centripetal_shift (bool): 是否生成 centripetal shift target.默认: False.

        Returns:
            dict: corner heatmap, corner offset, corner embedding, guiding shift
                and centripetal shift的拟合目标 . 包含以下键(倒数五个不一定存在):

                - topleft_heatmap (Tensor): 左上corner heatmap的拟合目标.
                - bottomright_heatmap (Tensor): 右下corner heatmap的拟合目标.
                - topleft_offset (Tensor): 左上corner offset的拟合目标.
                - bottomright_offset (Tensor): 右下corner offset的拟合目标.
                - corner_embedding (list[list[list[int]]]): corner offset的拟合目标.
                - topleft_guiding_shift (Tensor): 左上corner guiding shift的拟合目标.
                - bottomright_guiding_shift (Tensor): 右下corner guiding shift的拟合目标.
                - topleft_centripetal_shift (Tensor): 左上corner centripetal shift的拟合目标.
                - bottomright_centripetal_shift (Tensor): 右下corner centripetal shift的拟合目标.
        """
        batch_size, _, height, width = feat_shape
        img_h, img_w = img_shape[:2]

        width_ratio = float(width / img_w)
        height_ratio = float(height / img_h)

        # 初始化corner offset的拟合目标
        gt_tl_heatmap = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, height, width])
        gt_br_heatmap = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, height, width])
        gt_tl_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        gt_br_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])

        if with_corner_emb:
            match = []

        # Guiding shift 是一种从中心到角落的偏移
        if with_guiding_shift:
            gt_tl_guiding_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])
            gt_br_guiding_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])
        # Centripetal shift 也是一种偏移,从中心到角落并通过对数归一化.
        if with_centripetal_shift:
            gt_tl_centripetal_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])
            gt_br_centripetal_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])

        for batch_id in range(batch_size):
            # bs张图像的embedding target [[[t, l],[b, r],] * num_gt], ] * bs
            # t, l, b, r皆为int型.表示两个角点所属左上角索引
            corner_match = []
            for box_id in range(len(gt_labels[batch_id])):
                left, top, right, bottom = gt_bboxes[batch_id][box_id]
                center_x = (left + right) / 2.0
                center_y = (top + bottom) / 2.0
                label = gt_labels[batch_id][box_id]

                # 将gt的坐标缩放到特征图尺寸上
                scale_left = left * width_ratio
                scale_right = right * width_ratio
                scale_top = top * height_ratio
                scale_bottom = bottom * height_ratio
                scale_center_x = center_x * width_ratio
                scale_center_y = center_y * height_ratio

                # 获取gt缩放后的int型坐标
                left_idx = int(min(scale_left, width - 1))
                right_idx = int(min(scale_right, width - 1))
                top_idx = int(min(scale_top, height - 1))
                bottom_idx = int(min(scale_bottom, height - 1))

                # 生成corner heatmap
                scale_box_width = ceil(scale_right - scale_left)
                scale_box_height = ceil(scale_bottom - scale_top)
                radius = gaussian_radius([scale_box_height, scale_box_width],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                # 计算指定gt box的corner heatmap的高斯核,可能存在重合部分(取较大值)
                gt_tl_heatmap[batch_id, label] = gen_gaussian_target(
                    gt_tl_heatmap[batch_id, label], [left_idx, top_idx],
                    radius)
                gt_br_heatmap[batch_id, label] = gen_gaussian_target(
                    gt_br_heatmap[batch_id, label], [right_idx, bottom_idx],
                    radius)

                # 生成corner offset,在左上/右下的相对左上角的偏移.
                left_offset = scale_left - left_idx
                top_offset = scale_top - top_idx
                right_offset = scale_right - right_idx
                bottom_offset = scale_bottom - bottom_idx
                gt_tl_offset[batch_id, 0, top_idx, left_idx] = left_offset
                gt_tl_offset[batch_id, 1, top_idx, left_idx] = top_offset
                gt_br_offset[batch_id, 0, bottom_idx, right_idx] = right_offset
                gt_br_offset[batch_id, 1, bottom_idx, right_idx] = bottom_offset

                # 生成corner embedding
                if with_corner_emb:
                    corner_match.append([[top_idx, left_idx],
                                         [bottom_idx, right_idx]])
                # Generate guiding shift
                if with_guiding_shift:
                    gt_tl_guiding_shift[batch_id, 0, top_idx,
                                        left_idx] = scale_center_x - left_idx
                    gt_tl_guiding_shift[batch_id, 1, top_idx,
                                        left_idx] = scale_center_y - top_idx
                    gt_br_guiding_shift[batch_id, 0, bottom_idx,
                                        right_idx] = right_idx - scale_center_x
                    gt_br_guiding_shift[
                        batch_id, 1, bottom_idx,
                        right_idx] = bottom_idx - scale_center_y
                # Generate centripetal shift
                if with_centripetal_shift:
                    gt_tl_centripetal_shift[batch_id, 0, top_idx,
                                            left_idx] = log(scale_center_x -
                                                            scale_left)
                    gt_tl_centripetal_shift[batch_id, 1, top_idx,
                                            left_idx] = log(scale_center_y -
                                                            scale_top)
                    gt_br_centripetal_shift[batch_id, 0, bottom_idx,
                                            right_idx] = log(scale_right -
                                                             scale_center_x)
                    gt_br_centripetal_shift[batch_id, 1, bottom_idx,
                                            right_idx] = log(scale_bottom -
                                                             scale_center_y)

            if with_corner_emb:
                match.append(corner_match)

        target_result = dict(
            topleft_heatmap=gt_tl_heatmap,
            topleft_offset=gt_tl_offset,
            bottomright_heatmap=gt_br_heatmap,
            bottomright_offset=gt_br_offset)

        if with_corner_emb:
            target_result.update(corner_embedding=match)
        if with_guiding_shift:
            target_result.update(
                topleft_guiding_shift=gt_tl_guiding_shift,
                bottomright_guiding_shift=gt_br_guiding_shift)
        if with_centripetal_shift:
            target_result.update(
                topleft_centripetal_shift=gt_tl_centripetal_shift,
                bottomright_centripetal_shift=gt_br_centripetal_shift)

        return target_result

    @force_fp32()
    def loss(self,
             tl_heats,
             br_heats,
             tl_embs,
             br_embs,
             tl_offs,
             br_offs,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """计算head的loss.
            需要注意的是CornerNet默认使用HourglassNet-104,该主干网络的stack=2
            而CornerNet的Head部分在两个层级上的特征图上是都计算loss的,并且其在各自特征图上
            生成的target都是一致的(直接复制),但是在测试阶段只考虑最后一层特征图!!!


        Args:
            tl_heats (list[Tensor]): 所有层级上的左上 corner heatmaps,
                [[bs, num_classes, h, w],] * num_stack.
            br_heats (list[Tensor]): 所有层级上的右下 corner heatmaps,
                [[bs, num_classes, h, w],] * num_stack.
            tl_embs (list[Tensor]): 所有层级上的左上 corner embedding,
                [[bs, 1, h, w],] * num_stack.
            br_embs (list[Tensor]): 所有层级上的右下 corner embedding,
                [[bs, 1, h, w],] * num_stack.
            tl_offs (list[Tensor]): 所有层级上的左上 corner offset
                [[bs, 2, h, w],] * num_stack.
            br_offs (list[Tensor]): 所有层级上的右下 corner offset
                [[bs, 2, h, w],] * num_stack.
            gt_bboxes (list[Tensor]): [[num_gts, 4],] * bs, [x1,y1,x2,y2].
            gt_labels (list[Tensor]): [[num_gts,],] * bs
            img_metas (list[dict]): [dict(),] * bs.
            gt_bboxes_ignore (list[Tensor] | None): [[num_ignore_gts, 4],] * bs
                计算loss时可以忽略的哪些边界框.

        Returns:
            dict[str, Tensor]: loss字典。包含以下种类的loss:

                - det_loss (list[Tensor]): 所有层级的Corner keypoint loss.
                - pull_loss (list[Tensor]): Part one of AssociativeEmbedding
                  losses of all feature levels.
                - push_loss (list[Tensor]): Part two of AssociativeEmbedding
                  losses of all feature levels.
                - off_loss (list[Tensor]): 所有层级的Corner offset loss..
        """
        targets = self.get_targets(
            gt_bboxes,
            gt_labels,
            tl_heats[-1].shape,
            img_metas[0]['pad_shape'],
            with_corner_emb=self.with_corner_emb)
        mlvl_targets = [targets for _ in range(self.num_feat_levels)]
        det_losses, pull_losses, push_losses, off_losses = multi_apply(
            self.loss_single, tl_heats, br_heats, tl_embs, br_embs, tl_offs,
            br_offs, mlvl_targets)
        loss_dict = dict(det_loss=det_losses, off_loss=off_losses)
        if self.with_corner_emb:
            loss_dict.update(pull_loss=pull_losses, push_loss=push_losses)
        return loss_dict

    def loss_single(self, tl_hmp, br_hmp, tl_emb, br_emb, tl_off, br_off,
                    targets):
        """计算单层级上的损失.

        Args:
            tl_hmp (Tensor): 当前层级上的左上 corner heatmap[bs, num_classes, h, w].
            br_hmp (Tensor): 当前层级上的右下 corner heatmap[bs, num_classes, h, w].
            tl_emb (Tensor): 当前层级上的左上 corner embedding[bs, corner_emb_channels, h, w].
            br_emb (Tensor): 当前层级上的右下 corner embedding[bs, corner_emb_channels, h, w].
            tl_off (Tensor): 当前层级上的左上 corner offset[bs, corner_offset_channels, h, w].
            br_off (Tensor): 当前层级上的右下 corner offset[bs, corner_offset_channels, h, w].
            targets (dict): 由`get_targets`生成的三个Corner target .
                {
                    topleft_heatmap:[bs, num_classes, h, w].
                    topleft_offset:[bs, 2, h, w].
                    bottomright_heatmap:[bs, num_classes, h, w].
                    bottomright_offset:[bs, 2, h, w].
                    corner_embedding:[[[top_idx, left_idx],[bottom_idx, right_idx],] * num_gt] * bs
                }

        Returns:
            tuple[torch.Tensor]: head中不同分支的损失,包含以下:

                - det_loss (Tensor): Corner keypoint loss.
                - pull_loss (Tensor): AssociativeEmbedding loss中的pull loss.
                - push_loss (Tensor): AssociativeEmbedding loss中的push loss.
                - off_loss (Tensor): Corner offset loss.
        """
        gt_tl_hmp = targets['topleft_heatmap']
        gt_br_hmp = targets['bottomright_heatmap']
        gt_tl_off = targets['topleft_offset']
        gt_br_off = targets['bottomright_offset']
        gt_embedding = targets['corner_embedding']

        # Detection loss
        tl_det_loss = self.loss_heatmap(
            tl_hmp.sigmoid(),
            gt_tl_hmp,
            avg_factor=max(1,
                           gt_tl_hmp.eq(1).sum()))
        br_det_loss = self.loss_heatmap(
            br_hmp.sigmoid(),
            gt_br_hmp,
            avg_factor=max(1,
                           gt_br_hmp.eq(1).sum()))
        det_loss = (tl_det_loss + br_det_loss) / 2.0

        # AssociativeEmbedding loss
        if self.with_corner_emb and self.loss_embedding is not None:
            pull_loss, push_loss = self.loss_embedding(tl_emb, br_emb,
                                                       gt_embedding)
        else:
            pull_loss, push_loss = None, None

        # offset loss,分别代表corner在其所属grid区域上的横向与纵向的偏移∈[0, 1),单位:8px
        # 我们仅计算gt corner的位置的偏移损失.即在gt_tl/br_hmp中值为1的点(gt左上右下角的左上角坐标)
        # gt_tl_hmp.eq(1) -> [bs,num_class,h,w]任意位置上是否有gt,.sum(1)就是获取类别维度上有多少gt,
        # 也就是该张图像在指定位置中有多少gt,.gt(0)就是获取任意位置上是否含有gt box
        # tl_off_mask最终shape 为 [bs, 1, h, w].该值作为off_loss的权重
        tl_off_mask = gt_tl_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_tl_hmp)
        br_off_mask = gt_br_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_br_hmp)
        tl_off_loss = self.loss_offset(
            tl_off,
            gt_tl_off,
            tl_off_mask,
            avg_factor=max(1, tl_off_mask.sum()))
        br_off_loss = self.loss_offset(
            br_off,
            gt_br_off,
            br_off_mask,
            avg_factor=max(1, br_off_mask.sum()))

        off_loss = (tl_off_loss + br_off_loss) / 2.0

        return det_loss, pull_loss, push_loss, off_loss

    @force_fp32()
    def get_bboxes(self,
                   tl_heats,
                   br_heats,
                   tl_embs,
                   br_embs,
                   tl_offs,
                   br_offs,
                   img_metas,
                   rescale=False,
                   with_nms=True):
        """将模型输出(batch)转换为预测的box.注意,测试阶段只考虑最后一层特征图!!!

        Args:
            tl_heats (list[Tensor]): 左上 corner heatmaps
                [bs, num_classes, H, W] * num_level.
            br_heats (list[Tensor]): 右下 corner heatmaps
                [bs, num_classes, H, W] * num_level.
            tl_embs (list[Tensor]): 左上 corner embeddings
                [bs, corner_emb_channels, H, W] * num_level.
            br_embs (list[Tensor]): 右下 corner embeddings
                [bs, corner_emb_channels, H, W] * num_level.
            tl_offs (list[Tensor]): 左上 corner offsets
                [bs, 2, H, W] * num_level.
            br_offs (list[Tensor]): 右下 corner offsets
                [bs, 2, H, W] * num_level.
            img_metas (list[dict]): batch张图像元信息, [dict(),] * bs.
            rescale (bool): 如果为True, 则将预测box缩放回原始图像尺寸上.
            with_nms (bool): 如果为True, 在返回box前实行nms操作.
        """
        assert tl_heats[-1].shape[0] == br_heats[-1].shape[0] == len(img_metas)
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    tl_heats[-1][img_id:img_id + 1, :],
                    br_heats[-1][img_id:img_id + 1, :],
                    tl_offs[-1][img_id:img_id + 1, :],
                    br_offs[-1][img_id:img_id + 1, :],
                    img_metas[img_id],
                    tl_emb=tl_embs[-1][img_id:img_id + 1, :],
                    br_emb=br_embs[-1][img_id:img_id + 1, :],
                    rescale=rescale,
                    with_nms=with_nms))

        return result_list

    def _get_bboxes_single(self,
                           tl_heat,
                           br_heat,
                           tl_off,
                           br_off,
                           img_meta,
                           tl_emb=None,
                           br_emb=None,
                           tl_centripetal_shift=None,
                           br_centripetal_shift=None,
                           rescale=False,
                           with_nms=True):
        """将模型输出(单张图片)转换为预测的box.

        Args:
            tl_heat (Tensor): 左上corner heatmaps(最后一层,下同) [1, num_classes, H, W].
            br_heat (Tensor): 右下corner heatmaps [1, num_classes, H, W].
            tl_off (Tensor): 左上corner offset [1, 2, H, W].
            br_off (Tensor): 右下corner offset [1, 2, H, W].
            img_meta (dict): 当前图片的元信息.
            tl_emb (Tensor): 左上corner embedding [1, corner_emb_channels, H, W].
            br_emb (Tensor): 右下corner embedding [1, corner_emb_channels, H, W].
            tl_centripetal_shift: 左上corner centripetal shift [1, 2, H, W].
            br_centripetal_shift: 右下corner centripetal shift [1, 2, H, W].
            rescale (bool): 如果为True, 则将预测box缩放回原始图像尺寸上.默认: False.
            with_nms (bool): 如果为True, 在返回box前实行nms操作,默认: True.
        """
        if isinstance(img_meta, (list, tuple)):
            img_meta = img_meta[0]

        batch_bboxes, batch_scores, batch_clses = self.decode_heatmap(
            tl_heat=tl_heat.sigmoid(),
            br_heat=br_heat.sigmoid(),
            tl_off=tl_off,
            br_off=br_off,
            tl_emb=tl_emb,
            br_emb=br_emb,
            tl_centripetal_shift=tl_centripetal_shift,
            br_centripetal_shift=br_centripetal_shift,
            img_meta=img_meta,
            k=self.test_cfg.corner_topk,
            kernel=self.test_cfg.local_maximum_kernel,
            distance_threshold=self.test_cfg.distance_threshold)

        if rescale:
            batch_bboxes /= batch_bboxes.new_tensor(img_meta['scale_factor'])

        bboxes = batch_bboxes.view([-1, 4])
        scores = batch_scores.view(-1)
        clses = batch_clses.view(-1)

        detections = torch.cat([bboxes, scores.unsqueeze(-1)], -1)
        keepinds = (detections[:, -1] > -0.1)
        detections = detections[keepinds]
        labels = clses[keepinds]

        if with_nms:
            detections, labels = self._bboxes_nms(detections, labels,
                                                  self.test_cfg)

        return detections, labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            # 注意batch-nms,并非是将一整个batch图片的预测box同时进行nms
            # 而是对一张图片上所有类别的box进行nms,区分不同类别box的方法就是
            # 在不同类别box坐标上添加base_val*cls_id,base_val取自最大box坐标
            # cls_id为每个box所属类别id
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:,
                                                             -1].contiguous(),
                                       labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]

        return bboxes, labels

    def decode_heatmap(self,
                       tl_heat,
                       br_heat,
                       tl_off,
                       br_off,
                       tl_emb=None,
                       br_emb=None,
                       tl_centripetal_shift=None,
                       br_centripetal_shift=None,
                       img_meta=None,
                       k=100,
                       kernel=3,
                       distance_threshold=0.5,
                       num_dets=1000):
        """将模型输出(单张图片)转换为预测的box.
            需要注意的是,corner heatmap相当于每个corner的score,类似YOLO系列中的obj_score,
            首先得出合适的若干corner,因为CornerNet是corner-base.主要是枚举所有k*k种corner
            的组合方式,然后过各种限制对其过滤最后返回predict_box/score/label


        Args:
            tl_heat (Tensor): 左上corner heatmaps [1, num_classes, H, W].等价于各corner的obj_score
            br_heat (Tensor): 右下corner heatmaps [1, num_classes, H, W].
            tl_off (Tensor): 左上corner offset [1, 2, H, W].等价于各corner的相对偏移值,2代表横纵向
            br_off (Tensor): 右下corner offset [1, 2, H, W].
            tl_emb (Tensor | None): 左上corner embedding [1, corner_emb_channels, H, W].
            br_emb (Tensor | None): 右下corner embedding [1, corner_emb_channels, H, W].
            tl_centripetal_shift (Tensor | None): 左上corner centripetal shift
                [1, 2, H, W].
            br_centripetal_shift (Tensor | None): 右下corner centripetal shift
                [1, 2, H, W].
            img_meta (dict): 当前图片的元信息.
            k (int): 从heatmap中截取前k个corner keypoint.
            kernel (int): 用于提取局部最大像素的最大池化核大小.
            distance_threshold (float): 距离阈值. 左上和右下的特征距离小于该阈值的
                一对corner将被视为来自同一gt box.
            num_dets (int): 进行 nms 之后的最大box数量.

        Returns:
            tuple[torch.Tensor]: CornerHead 的解码输出, 包含以下张量-:

            - bboxes (Tensor): 每个box的两个corner坐标(x,y).
            - scores (Tensor): 每个box的score.
            - clses (Tensor): 每个box的类别.
        """
        with_embedding = tl_emb is not None and br_emb is not None
        with_centripetal_shift = (
            tl_centripetal_shift is not None
            and br_centripetal_shift is not None)
        assert with_embedding + with_centripetal_shift == 1
        batch, _, height, width = tl_heat.size()
        if torch.onnx.is_in_onnx_export():
            inp_h, inp_w = img_meta['pad_shape_for_onnx'][:2]
        else:
            # 在CornerNet系列网络中,该值代表做完MultiScaleFlipAug增强操作后的shape
            # 其在训练与测试阶段的具体操作不同,有关剪裁区域这里不再描述,(训练时)该操作后的
            # 图像尺寸会生成到指定crop_size,(测试时)则生成图像尺寸向上兼容指定像素的倍数.
            # 而这里的pad_shape就代表生成图像的尺寸.
            inp_h, inp_w, _ = img_meta['pad_shape']

        # 在heatmap上执行nms
        tl_heat = get_local_maximum(tl_heat, kernel=kernel)
        br_heat = get_local_maximum(br_heat, kernel=kernel)

        # [bs, k]
        tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = get_topk_from_heatmap(
            tl_heat, k=k)
        br_scores, br_inds, br_clses, br_ys, br_xs = get_topk_from_heatmap(
            br_heat, k=k)

        # 我们在这里使用repeat而非expand,因为expand是一个浅拷贝函数.
        # 因此有时会导致意外的测试结果.与repeat相比,在测试期间使用expand将降低约10%的mAP.
        # 第一个k代表tl维度,第二个k代表br维度,所以tl需要向右复制k列,br需要向下复制k行
        # 所以对于tl_ys/xs和br_ys/xs来说会有不同的view与repeat操作
        # 直白一点就是,网络这里筛选(通过corner heatmap)出k个左上/右下坐标后.
        # 模型还并不知道这k个左上和k个右下坐标如何对应,既然不知道索性就用暴力法,直接生成k*k种可能
        tl_ys = tl_ys.view(batch, k, 1).repeat(1, 1, k)
        tl_xs = tl_xs.view(batch, k, 1).repeat(1, 1, k)
        br_ys = br_ys.view(batch, 1, k).repeat(1, k, 1)
        br_xs = br_xs.view(batch, 1, k).repeat(1, k, 1)

        # [bs, k, 2],tl_off不需要显示进行repeat,它依靠广播机制也可实现repeat操作
        tl_off = transpose_and_gather_feat(tl_off, tl_inds)
        tl_off = tl_off.view(batch, k, 1, 2)
        br_off = transpose_and_gather_feat(br_off, br_inds)
        br_off = br_off.view(batch, 1, k, 2)

        # 例,score高的corner的x坐标+该位置的修正系数=在该特征图下的x坐标,单位:8px.下同
        tl_xs = tl_xs + tl_off[..., 0]
        tl_ys = tl_ys + tl_off[..., 1]
        br_xs = br_xs + br_off[..., 0]
        br_ys = br_ys + br_off[..., 1]

        if with_centripetal_shift:
            tl_centripetal_shift = transpose_and_gather_feat(
                tl_centripetal_shift, tl_inds).view(batch, k, 1, 2).exp()
            br_centripetal_shift = transpose_and_gather_feat(
                br_centripetal_shift, br_inds).view(batch, 1, k, 2).exp()

            tl_ctxs = tl_xs + tl_centripetal_shift[..., 0]
            tl_ctys = tl_ys + tl_centripetal_shift[..., 1]
            br_ctxs = br_xs - br_centripetal_shift[..., 0]
            br_ctys = br_ys - br_centripetal_shift[..., 1]

        # 将前k个corner坐标缩放回Pad后的图像尺寸上
        tl_xs *= (inp_w / width)
        tl_ys *= (inp_h / height)
        br_xs *= (inp_w / width)
        br_ys *= (inp_h / height)

        if with_centripetal_shift:
            tl_ctxs *= (inp_w / width)
            tl_ctys *= (inp_h / height)
            br_ctxs *= (inp_w / width)
            br_ctys *= (inp_h / height)

        x_off, y_off = 0, 0  # 没有剪裁时
        if not torch.onnx.is_in_onnx_export():
            # 因为 `RandomCenterCropPad` 是在 CPU 上使用 numpy 完成的,并且在导出到
            # ONNX时它不是动态可跟踪的,因此 'border' 不会作为 'img_meta'中的键出现.
            # 作为一个临时解决方案,在完成导出到ONNX之后,我们将这部分移动到模型后处理中解决.
            # 该部分在`mmdet/core/export/model_wrappers.py中`. 尽管pytorch和导出的
            # onnx 模型之间存在差异,但可以被忽略,因为它们之间实现了相当的性能(例如在没有
            # test-time flip的CornerNet上,他们的COCO val2017 mAP分别为40.4与40.6)
            # img_meta['border']代表生成图像的粘贴区域, [top, bottom, left, right],
            # 这四个值分别表示在y,x,y,x坐标上的分割直线,相对于生成图像尺寸.
            # 这也是get_bboxes的参数rescale为False的原因,因为该操作在这里执行了.
            if 'border' in img_meta:
                x_off = img_meta['border'][2]
                y_off = img_meta['border'][0]

        # 将网络在生成图像上预测的box减去padding部分的坐标,以适应原始图像尺寸.
        tl_xs -= x_off
        tl_ys -= y_off
        br_xs -= x_off
        br_ys -= y_off

        # 限制两个corner的 x y坐标(>=0)
        zeros = tl_xs.new_zeros(*tl_xs.size())
        tl_xs = torch.where(tl_xs > 0.0, tl_xs, zeros)
        tl_ys = torch.where(tl_ys > 0.0, tl_ys, zeros)
        br_xs = torch.where(br_xs > 0.0, br_xs, zeros)
        br_ys = torch.where(br_ys > 0.0, br_ys, zeros)

        # [bs, k, k, 4]
        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)
        area_bboxes = ((br_xs - tl_xs) * (br_ys - tl_ys)).abs()

        if with_centripetal_shift:
            tl_ctxs -= x_off
            tl_ctys -= y_off
            br_ctxs -= x_off
            br_ctys -= y_off

            tl_ctxs *= tl_ctxs.gt(0.0).type_as(tl_ctxs)
            tl_ctys *= tl_ctys.gt(0.0).type_as(tl_ctys)
            br_ctxs *= br_ctxs.gt(0.0).type_as(br_ctxs)
            br_ctys *= br_ctys.gt(0.0).type_as(br_ctys)

            ct_bboxes = torch.stack((tl_ctxs, tl_ctys, br_ctxs, br_ctys),
                                    dim=3)
            area_ct_bboxes = ((br_ctxs - tl_ctxs) * (br_ctys - tl_ctys)).abs()

            rcentral = torch.zeros_like(ct_bboxes)
            # magic nums from paper section 4.1
            mu = torch.ones_like(area_bboxes) / 2.4
            mu[area_bboxes > 3500] = 1 / 2.1  # large bbox have smaller mu

            bboxes_center_x = (bboxes[..., 0] + bboxes[..., 2]) / 2
            bboxes_center_y = (bboxes[..., 1] + bboxes[..., 3]) / 2
            rcentral[..., 0] = bboxes_center_x - mu * (bboxes[..., 2] -
                                                       bboxes[..., 0]) / 2
            rcentral[..., 1] = bboxes_center_y - mu * (bboxes[..., 3] -
                                                       bboxes[..., 1]) / 2
            rcentral[..., 2] = bboxes_center_x + mu * (bboxes[..., 2] -
                                                       bboxes[..., 0]) / 2
            rcentral[..., 3] = bboxes_center_y + mu * (bboxes[..., 3] -
                                                       bboxes[..., 1]) / 2
            area_rcentral = ((rcentral[..., 2] - rcentral[..., 0]) *
                             (rcentral[..., 3] - rcentral[..., 1])).abs()
            dists = area_ct_bboxes / area_rcentral

            tl_ctx_inds = (ct_bboxes[..., 0] <= rcentral[..., 0]) | (
                ct_bboxes[..., 0] >= rcentral[..., 2])
            tl_cty_inds = (ct_bboxes[..., 1] <= rcentral[..., 1]) | (
                ct_bboxes[..., 1] >= rcentral[..., 3])
            br_ctx_inds = (ct_bboxes[..., 2] <= rcentral[..., 0]) | (
                ct_bboxes[..., 2] >= rcentral[..., 2])
            br_cty_inds = (ct_bboxes[..., 3] <= rcentral[..., 1]) | (
                ct_bboxes[..., 3] >= rcentral[..., 3])

        if with_embedding:  # 获取k个tl_emb与k个br_emb的绝对差距
            tl_emb = transpose_and_gather_feat(tl_emb, tl_inds)
            tl_emb = tl_emb.view(batch, k, 1)
            br_emb = transpose_and_gather_feat(br_emb, br_inds)
            br_emb = br_emb.view(batch, 1, k)
            dists = torch.abs(tl_emb - br_emb)

        tl_scores = tl_scores.view(batch, k, 1).repeat(1, 1, k)
        br_scores = br_scores.view(batch, 1, k).repeat(1, k, 1)

        scores = (tl_scores + br_scores) / 2  # 所有可能的box([bs,k,k])的score

        # 如果左上和右下表示某个box的话,那么它们两应该属于同一类别.否则不属于同一box
        tl_clses = tl_clses.view(batch, k, 1).repeat(1, 1, k)
        br_clses = br_clses.view(batch, 1, k).repeat(1, k, 1)
        cls_inds = (tl_clses != br_clses)

        # 过滤左上和右下的emb差距大于指定阈值的corner
        dist_inds = dists > distance_threshold

        # 过滤宽高小于等于0的.
        width_inds = (br_xs <= tl_xs)
        height_inds = (br_ys <= tl_ys)

        # 我们在这里使用 `torch.where`来代替`scores[cls_inds]`.
        # Since only 1-D indices with type 'tensor(bool)' are supported
        # when exporting to ONNX, any other bool indices with more dimensions
        # (e.g. 2-D bool tensor) as input parameter in node is invalid
        # 因为在导出到 ONNX 时仅支持类型为“tensor(bool)”的一维索引,因此任何其他具有
        # 更多维度的bool indices(例如二维bool tensor)作为节点中的输入参数都是无效的
        negative_scores = -1 * torch.ones_like(scores)
        scores = torch.where(cls_inds, negative_scores, scores)
        scores = torch.where(width_inds, negative_scores, scores)
        scores = torch.where(height_inds, negative_scores, scores)
        scores = torch.where(dist_inds, negative_scores, scores)

        if with_centripetal_shift:
            scores[tl_ctx_inds] = -1
            scores[tl_cty_inds] = -1
            scores[br_ctx_inds] = -1
            scores[br_cty_inds] = -1

        # [bs, k, k] -> [bs, k*k]  仅排序选前num_det个,此时还没通过score_thr来进行过滤
        scores = scores.view(batch, -1)
        scores, inds = torch.topk(scores, num_dets)
        # [bs, num_dets, 1]
        scores = scores.unsqueeze(2)

        bboxes = bboxes.view(batch, -1, 4)
        bboxes = gather_feat(bboxes, inds)

        clses = tl_clses.contiguous().view(batch, -1, 1)
        clses = gather_feat(clses, inds).float()
        # 理论上最大shape为[bs,num_det,4/1/1], box/score/cls
        return bboxes, scores, clses

    def onnx_export(self,
                    tl_heats,
                    br_heats,
                    tl_embs,
                    br_embs,
                    tl_offs,
                    br_offs,
                    img_metas,
                    rescale=False,
                    with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_embs (list[Tensor]): Top-left corner embeddings for each level
                with shape (N, corner_emb_channels, H, W).
            br_embs (list[Tensor]): Bottom-right corner embeddings for each
                level with shape (N, corner_emb_channels, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor]: First tensor bboxes with shape
            [N, num_det, 5], 5 arrange as (x1, y1, x2, y2, score)
            and second element is class labels of shape [N, num_det].
        """
        assert tl_heats[-1].shape[0] == br_heats[-1].shape[0] == len(
            img_metas) == 1
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    tl_heats[-1][img_id:img_id + 1, :],
                    br_heats[-1][img_id:img_id + 1, :],
                    tl_offs[-1][img_id:img_id + 1, :],
                    br_offs[-1][img_id:img_id + 1, :],
                    img_metas[img_id],
                    tl_emb=tl_embs[-1][img_id:img_id + 1, :],
                    br_emb=br_embs[-1][img_id:img_id + 1, :],
                    rescale=rescale,
                    with_nms=with_nms))

        detections, labels = result_list[0]
        # batch_size 1 here, [1, num_det, 5], [1, num_det]
        return detections.unsqueeze(0), labels.unsqueeze(0)
