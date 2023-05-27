# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmdet.core import InstanceData, mask_matrix_nms, multi_apply
from mmdet.core.utils import center_of_mass, generate_coordinate
from mmdet.models.builder import HEADS
from .solo_head import SOLOHead


class MaskFeatModule(BaseModule):
    """SOLOv2 mask feature map branch used in `SOLOv2: Dynamic and Fast
    Instance Segmentation. <https://arxiv.org/pdf/2003.10152>`_

    Args:
        in_channels (int): 输入特征图中的通道数.
        feat_channels (int): mask feature map分支的隐藏通道数.
        start_level (int): 来自 RPN 的起始特征图索引,将从该索引开始用于mask feature map.
        end_level (int): 来自 RPN 的截至特征图索引,将到该索引为止用于mask feature map.
        out_channels (int): mask feature map branch的输出通道数. This is the channel count of the mask
             feature map that to be dynamically convolved with the predicted
             kernel.
        mask_stride (int): mask feature map 输出的下采样因子. 默认: 4.
        conv_cfg (dict): conv层的配置. 默认: None.
        norm_cfg (dict): norm层的配置. 默认: None.
        init_cfg (dict or list[dict], optional): 权重的初始化配置.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 start_level,
                 end_level,
                 out_channels,
                 mask_stride=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01)]):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.start_level = start_level
        self.end_level = end_level
        self.mask_stride = mask_stride
        assert start_level >= 0 and end_level >= start_level
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()
        self.fp16_enabled = False

    def _init_layers(self):
        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                convs_per_level.add_module(
                    f'conv{i}',
                    ConvModule(
                        self.in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=False))
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    if i == self.end_level:
                        chn = self.in_channels + 2
                    else:
                        chn = self.in_channels
                    convs_per_level.add_module(
                        f'conv{j}',
                        ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            inplace=False))
                    convs_per_level.add_module(
                        f'upsample{j}',
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False))
                    continue

                convs_per_level.add_module(
                    f'conv{j}',
                    ConvModule(
                        self.feat_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=False))
                convs_per_level.add_module(
                    f'upsample{j}',
                    nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False))

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = ConvModule(
            self.feat_channels,
            self.out_channels,
            1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

    @auto_fp16()
    def forward(self, feats):
        inputs = feats[self.start_level:self.end_level + 1]
        assert len(inputs) == (self.end_level - self.start_level + 1)
        feature_add_all_level = self.convs_all_levels[0](inputs[0])
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == len(inputs) - 1:
                coord_feat = generate_coordinate(input_p.size(),
                                                 input_p.device)
                input_p = torch.cat([input_p, coord_feat], 1)

            # fix runtime error of "+=" inplace operation in PyTorch 1.10
            feature_add_all_level = feature_add_all_level + \
                self.convs_all_levels[i](input_p)

        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred


@HEADS.register_module()
class SOLOV2Head(SOLOHead):
    """SOLOv2 mask head used in `SOLOv2: Dynamic and Fast Instance
    Segmentation. <https://arxiv.org/pdf/2003.10152>`_

    Args:
        mask_feature_head (dict): SOLOv2MaskFeatHead的配置项.
        dynamic_conv_size (int): Dynamic Conv kernel size. Default: 1.
        dcn_cfg (dict): kernel_conv 和 cls_conv中的DCN配置项.默认: None.
        dcn_apply_to_all_conv (bool): 在kernel_conv和cls_conv的每一层都使用dcn,
            还是只在最后一层使用,在SOLOv2-Light上为False. 默认: True.
        init_cfg (dict or list[dict], optional): 权重初始化配置.
    """

    def __init__(self,
                 *args,
                 mask_feature_head,
                 dynamic_conv_size=1,
                 dcn_cfg=None,
                 dcn_apply_to_all_conv=True,
                 init_cfg=[
                     dict(type='Normal', layer='Conv2d', std=0.01),
                     dict(
                         type='Normal',
                         std=0.01,
                         bias_prob=0.01,
                         override=dict(name='conv_cls'))
                 ],
                 **kwargs):
        assert dcn_cfg is None or isinstance(dcn_cfg, dict)
        self.dcn_cfg = dcn_cfg
        self.with_dcn = dcn_cfg is not None
        self.dcn_apply_to_all_conv = dcn_apply_to_all_conv
        self.dynamic_conv_size = dynamic_conv_size
        mask_out_channels = mask_feature_head.get('out_channels')
        self.kernel_out_channels = \
            mask_out_channels * self.dynamic_conv_size * self.dynamic_conv_size

        super().__init__(*args, init_cfg=init_cfg, **kwargs)

        # 更新mask_feature_head的in_channels, 不理解这段代码有何意义
        if mask_feature_head.get('in_channels', None) is not None:
            if mask_feature_head.in_channels != self.in_channels:
                warnings.warn('The `in_channels` of SOLOv2MaskFeatHead and '
                              'SOLOv2Head should be same, changing '
                              'mask_feature_head.in_channels to '
                              f'{self.in_channels}')
                mask_feature_head.update(in_channels=self.in_channels)
        else:
            mask_feature_head.update(in_channels=self.in_channels)

        self.mask_feature_head = MaskFeatModule(**mask_feature_head)
        self.mask_stride = self.mask_feature_head.mask_stride
        self.fp16_enabled = False

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        conv_cfg = None
        for i in range(self.stacked_convs):
            if self.with_dcn:
                # 是否在整个head区域都适用DCN还是仅在最后一层卷积适用
                if self.dcn_apply_to_all_conv:
                    conv_cfg = self.dcn_cfg
                elif i == self.stacked_convs - 1:
                    # light head
                    conv_cfg = self.dcn_cfg

            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

        self.conv_kernel = nn.Conv2d(
            self.feat_channels, self.kernel_out_channels, 3, padding=1)

    @auto_fp16()
    def forward(self, feats):
        assert len(feats) == self.num_levels
        # mask_feature_head默认配置下仅利用了前四层,[bs, out_c, batch_h//4, batch_w//4]
        mask_feats = self.mask_feature_head(feats)
        feats = self.resize_feats(feats)
        mlvl_kernel_preds = []
        mlvl_cls_preds = []
        for i in range(self.num_levels):
            ins_kernel_feat = feats[i]
            # ins branch
            # concat coord
            coord_feat = generate_coordinate(ins_kernel_feat.size(),
                                             ins_kernel_feat.device)
            ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

            # kernel branch
            kernel_feat = ins_kernel_feat
            kernel_feat = F.interpolate(
                kernel_feat,
                size=self.num_grids[i],
                mode='bilinear',
                align_corners=False)

            cate_feat = kernel_feat[:, :-2, :, :]

            kernel_feat = kernel_feat.contiguous()
            for i, kernel_conv in enumerate(self.kernel_convs):
                kernel_feat = kernel_conv(kernel_feat)
            kernel_pred = self.conv_kernel(kernel_feat)

            # cate branch
            cate_feat = cate_feat.contiguous()
            for i, cls_conv in enumerate(self.cls_convs):
                cate_feat = cls_conv(cate_feat)
            cate_pred = self.conv_cls(cate_feat)

            mlvl_kernel_preds.append(kernel_pred)
            mlvl_cls_preds.append(cate_pred)

        return mlvl_kernel_preds, mlvl_cls_preds, mask_feats

    def _get_targets_single(self,
                            gt_bboxes,
                            gt_labels,
                            gt_masks,
                            featmap_size=None):
        """计算单张图像的拟合目标.

        Args:
            gt_bboxes (Tensor): [num_gts, 4].
            gt_labels (Tensor): [num_gts, ].
            gt_masks (Tensor): [num_gts, pad_h, pad_w].
            featmap_size (:obj:`torch.size`): [batch_h//4, batch_w//4]. 默认: None.
                注意SOLOv1中,该参数是[[h, w], ] * nl.即v1中不同层级的mask_target尺寸是不同的
                而v2中是固定为[batch_h//4, batch_w//4],因为Head部分的mask输出是将2~4层级
                分别上采样至第1层级尺寸[batch_h//4, batch_w//4]再相加到一起的再通过
                conv1x1_GN_ReLU得到的.所以各层上生成的target_mask尺寸都是一致的.

        Returns:
            Tuple: 通常返回一个包含拟合目标的元组.

                - mlvl_pos_mask_targets (list[Tensor]): 所有层级上正样本的target_mask
                  [[num_pos_per_img_lvl, batch_h//4, batch_w//4], ] * nl.
                - mlvl_labels (list[Tensor]): 所有层级上正样本的target_cls
                  [[num_grid, num_grid], ] * nl.
                - mlvl_pos_masks  (list[Tensor]): bool值,正样本区域为True(0.2倍gt范围)
                   [[num_grid**2, ], ] * nl.
                - mlvl_pos_indexes  (list[list]): 所有层级上正样本区域的一维索引
                  [[num_pos_per_img_lvl, ], ] * nl.
        """

        device = gt_labels.device
        gt_areas = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) *
                              (gt_bboxes[:, 3] - gt_bboxes[:, 1]))

        mlvl_pos_mask_targets = []
        mlvl_pos_indexes = []
        mlvl_labels = []
        mlvl_pos_masks = []
        for (lower_bound, upper_bound), num_grid \
                in zip(self.scale_ranges, self.num_grids):
            mask_target = []
            # 前景: [0, nc), 背景: nc
            pos_index = []
            labels = torch.zeros([num_grid, num_grid],
                                 dtype=torch.int64,
                                 device=device) + self.num_classes
            pos_mask = torch.zeros([num_grid**2],
                                   dtype=torch.bool,
                                   device=device)

            gt_inds = ((gt_areas >= lower_bound) &
                       (gt_areas <= upper_bound)).nonzero().flatten()
            if len(gt_inds) == 0:
                mlvl_pos_mask_targets.append(
                    torch.zeros([0, featmap_size[0], featmap_size[1]],
                                dtype=torch.uint8,
                                device=device))
                mlvl_labels.append(labels)
                mlvl_pos_masks.append(pos_mask)
                mlvl_pos_indexes.append([])
                continue
            hit_gt_bboxes = gt_bboxes[gt_inds]
            hit_gt_labels = gt_labels[gt_inds]
            hit_gt_masks = gt_masks[gt_inds, ...]

            pos_w_ranges = 0.5 * (hit_gt_bboxes[:, 2] -
                                  hit_gt_bboxes[:, 0]) * self.pos_scale
            pos_h_ranges = 0.5 * (hit_gt_bboxes[:, 3] -
                                  hit_gt_bboxes[:, 1]) * self.pos_scale

            # Make sure hit_gt_masks has a value
            valid_mask_flags = hit_gt_masks.sum(dim=-1).sum(dim=-1) > 0

            for gt_mask, gt_label, pos_h_range, pos_w_range, \
                valid_mask_flag in \
                    zip(hit_gt_masks, hit_gt_labels, pos_h_ranges,
                        pos_w_ranges, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (featmap_size[0] * self.mask_stride,
                                  featmap_size[1] * self.mask_stride)
                center_h, center_w = center_of_mass(gt_mask)

                coord_w = int(
                    (center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int(
                    (center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(
                    0,
                    int(((center_h - pos_h_range) / upsampled_size[0]) //
                        (1. / num_grid)))
                down_box = min(
                    num_grid - 1,
                    int(((center_h + pos_h_range) / upsampled_size[0]) //
                        (1. / num_grid)))
                left_box = max(
                    0,
                    int(((center_w - pos_w_range) / upsampled_size[1]) //
                        (1. / num_grid)))
                right_box = min(
                    num_grid - 1,
                    int(((center_w + pos_w_range) / upsampled_size[1]) //
                        (1. / num_grid)))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                labels[top:(down + 1), left:(right + 1)] = gt_label
                # ins
                gt_mask = np.uint8(gt_mask.cpu().numpy())
                # Follow the original implementation, F.interpolate is
                # different from cv2 and opencv
                gt_mask = mmcv.imrescale(gt_mask, scale=1. / self.mask_stride)
                gt_mask = torch.from_numpy(gt_mask).to(device=device)

                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        index = int(i * num_grid + j)
                        this_mask_target = torch.zeros(
                            [featmap_size[0], featmap_size[1]],
                            dtype=torch.uint8,
                            device=device)
                        this_mask_target[:gt_mask.shape[0], :gt_mask.
                                         shape[1]] = gt_mask
                        mask_target.append(this_mask_target)
                        pos_mask[index] = True
                        pos_index.append(index)
            if len(mask_target) == 0:
                mask_target = torch.zeros(
                    [0, featmap_size[0], featmap_size[1]],
                    dtype=torch.uint8,
                    device=device)
            else:
                mask_target = torch.stack(mask_target, 0)
            mlvl_pos_mask_targets.append(mask_target)
            mlvl_labels.append(labels)
            mlvl_pos_masks.append(pos_mask)
            mlvl_pos_indexes.append(pos_index)
        return (mlvl_pos_mask_targets, mlvl_labels, mlvl_pos_masks,
                mlvl_pos_indexes)

    @force_fp32(apply_to=('mlvl_kernel_preds', 'mlvl_cls_preds', 'mask_feats'))
    def loss(self,
             mlvl_kernel_preds,
             mlvl_cls_preds,
             mask_feats,
             gt_labels,
             gt_masks,
             img_metas,
             gt_bboxes=None,
             **kwargs):
        """Calculate the loss of total batch.

        Args:
            mlvl_kernel_preds (list[Tensor]): 多层级的动态卷积输出.
                [[bs, kernel_c, num_grid, num_grid], ] * nl, 其中kernel_c为
                mask_out_c * dynamic_conv_size**2
            mlvl_cls_preds (list[Tensor]): 多层级的cls输出.
                [[bs, nc, num_grid, num_grid], ] * nl
            mask_feats (Tensor): 该值结合正样本对应的pred_kernel生成最终的pred_mask
                [bs, mask_out_c, batch_h//4, batch_w//4].
            gt_labels (list[Tensor]): [[num_gt, ], ] * bs
            gt_masks (list[Tensor]): [[num_gt, pad_h, pad_w], ] * bs.
            img_metas (list[dict]): [dict(), ] * bs.
            gt_bboxes (list[Tensor]): [[num_gt, 4], ] * bs. 默认: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_size = mask_feats.size()[-2:]
        # [[[num_pos_per_img_lvl, batch_h//4, batch_w//4], ] * nl], ] * bs. 正样本的target_mask
        # [[[num_grid, num_grid], ] * nl], ] * bs. 样本的target_cls(默认值为nc)
        # [[[num_grid**2, ], ] * nl], ] * bs. bool值,正样本区域为True(0.2倍gt范围)
        # [[[num_pos_per_img_lvl, ], ] * nl], ] * bs. 正样本区域的一维索引, ∈[0,h*w)
        pos_mask_targets, labels, pos_masks, pos_indexes = multi_apply(
            self._get_targets_single,
            gt_bboxes,
            gt_labels,
            gt_masks,
            featmap_size=featmap_size)
        # -> [[bs * num_pos_per_img_lvl, batch_h//4, batch_w//4], ] * nl
        mlvl_mask_targets = [
            torch.cat(lvl_mask_targets, 0)
            for lvl_mask_targets in zip(*pos_mask_targets)
        ]
        # 最终成为:[[[kernel_c, num_pos_per_img_lvl], ] * bs, ] * nl
        mlvl_pos_kernel_preds = []
        # 因为mlvl_kernel_preds是[l0, l1...], 而zip(*pos_indexes)->[l0, l1...]
        # 所以外层的zip就把这二者合到一起了. l0, l1...代表不同层级的数据.
        for lvl_kernel_preds, lvl_pos_indexes in zip(mlvl_kernel_preds,
                                                     zip(*pos_indexes)):
            lvl_pos_kernel_preds = []
            # 在单层级上循环不同img上的pred_kernel, pos_index
            for img_lvl_kernel_preds, img_lvl_pos_indexes in zip(
                    lvl_kernel_preds, lvl_pos_indexes):
                # [kernel_c, num_grid, num_grid] -> [kernel_c, num_grid**2]
                # 获取正样本区域位置所对应的pred_kernel
                img_lvl_pos_kernel_preds = img_lvl_kernel_preds.view(
                    img_lvl_kernel_preds.shape[0], -1)[:, img_lvl_pos_indexes]
                lvl_pos_kernel_preds.append(img_lvl_pos_kernel_preds)
            mlvl_pos_kernel_preds.append(lvl_pos_kernel_preds)

        # 生成多层级上的pred_mask
        mlvl_mask_preds = []
        for lvl_pos_kernel_preds in mlvl_pos_kernel_preds:
            lvl_mask_preds = []
            for img_id, img_lvl_pos_kernel_pred in enumerate(
                    lvl_pos_kernel_preds):
                # 如果当前层级当前图像上没有正样本,则跳过
                if img_lvl_pos_kernel_pred.size()[-1] == 0:
                    continue
                # [1, mask_out_c, batch_h//4, batch_w//4]. 在切片操作上[i]等价于i:i+1
                img_mask_feats = mask_feats[[img_id]]
                h, w = img_mask_feats.shape[-2:]
                # 正样本个数
                num_kernel = img_lvl_pos_kernel_pred.shape[1]
                # F.conv2d(input, weight, stride=1), 此处相当于对输入做了一个s=1的卷积操作
                # 这个卷积的核大小是固定的,输出维度等于正样本数量,具体参数则是网络预测的
                # 实际上网络所有层级上num_grid**2都预测了,最终是挑选正样本位置的预测
                # 再将其reshape成与img_mask_feats适配的shape与其进行卷积操作.
                # 最终view得到[1*num_pos_per_img_lvl, batch_h//4, batch_w//4]
                img_lvl_mask_pred = F.conv2d(
                    img_mask_feats,
                    # [kernel_c, num_pos_per_img_lvl] -> [num_pos_per_img_lvl, kernel_c]
                    # [num_pos_per_img_lvl, mask_out_c, dynamic_conv_size, dynamic_conv_size]
                    img_lvl_pos_kernel_pred.permute(1, 0).view(
                        num_kernel, -1, self.dynamic_conv_size,
                        self.dynamic_conv_size),
                    stride=1).view(-1, h, w)
                lvl_mask_preds.append(img_lvl_mask_pred)
            if len(lvl_mask_preds) == 0:
                lvl_mask_preds = None
            else:
                lvl_mask_preds = torch.cat(lvl_mask_preds, 0)
            # [[bs * num_pos_per_img_lvl, batch_h//4, batch_w//4], ] * nl
            mlvl_mask_preds.append(lvl_mask_preds)
        # dice loss, 统计batch张数据所有层级上正样本数量
        num_pos = 0
        for img_pos_masks in pos_masks:
            for lvl_img_pos_masks in img_pos_masks:
                num_pos += lvl_img_pos_masks.count_nonzero()

        loss_mask = []
        # mlvl_mask_targets: [[bs*num_pos_per_img_lvl, batch_h//4, batch_w//4], ] * nl
        for lvl_mask_preds, lvl_mask_targets in zip(mlvl_mask_preds,
                                                    mlvl_mask_targets):
            if lvl_mask_preds is None:  # 当前层级没有正样本
                continue
            loss_mask.append(
                self.loss_mask(
                    lvl_mask_preds,
                    lvl_mask_targets,
                    reduction_override='none'))
        if num_pos > 0:
            loss_mask = torch.cat(loss_mask).sum() / num_pos
        else:
            loss_mask = torch.cat(loss_mask).mean()

        # target_cls -> [[bs * num_grid**2, ], ] * nl
        flatten_labels = [
            torch.cat(
                [img_lvl_labels.flatten() for img_lvl_labels in lvl_labels])
            for lvl_labels in zip(*labels)
        ]
        flatten_labels = torch.cat(flatten_labels)
        # pred_cls -> [[bs*num_grid**2, nc], ] * nl
        flatten_cls_preds = [
            lvl_cls_preds.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for lvl_cls_preds in mlvl_cls_preds
        ]
        flatten_cls_preds = torch.cat(flatten_cls_preds)

        loss_cls = self.loss_cls(
            flatten_cls_preds, flatten_labels, avg_factor=num_pos + 1)
        return dict(loss_mask=loss_mask, loss_cls=loss_cls)

    @force_fp32(
        apply_to=('mlvl_kernel_preds', 'mlvl_cls_scores', 'mask_feats'))
    def get_results(self, mlvl_kernel_preds, mlvl_cls_scores, mask_feats,
                    img_metas, **kwargs):
        """Get multi-image mask results.

        Args:
            mlvl_kernel_preds (list[Tensor]): 多层级的动态卷积输出.利用它与mask_feat
                结合生成最终的pred_mask.[[bs, kernel_c, num_grid, num_grid], ] * nl
                其中kernel_c为mask_out_c * dynamic_conv_size**2
            mlvl_cls_scores (list[Tensor]): 多层级的cls输出.
                [[bs, nc, num_grid, num_grid], ] * nl
            mask_feats (Tensor): 该值结合mlvl_kernel_preds生成最终的pred_mask
                [bs, mask_out_c, batch_h//4, batch_w//4].
            img_metas (list[dict]): [dict(), ] * bs. batch幅图像元信息.

        Returns:
            list[:obj:`InstanceData`]: 处理后的batch幅图像分割结果.
            每个都是`InstanceData`数据结构.通常包含以下键.

                - scores (Tensor): pred_cls_score, [num_instance, ].
                - labels (Tensor): pred_cls_ind, [num_instance, ].
                - masks (Tensor): pred_mask, [num_instances, h, w].
        """
        num_levels = len(mlvl_cls_scores)
        assert len(mlvl_kernel_preds) == len(mlvl_cls_scores)

        for lvl in range(num_levels):
            cls_scores = mlvl_cls_scores[lvl]
            cls_scores = cls_scores.sigmoid()
            local_max = F.max_pool2d(cls_scores, 2, stride=1, padding=1)
            keep_mask = local_max[:, :, :-1, :-1] == cls_scores
            cls_scores = cls_scores * keep_mask
            mlvl_cls_scores[lvl] = cls_scores.permute(0, 2, 3, 1)

        result_list = []
        for img_id in range(len(img_metas)):
            img_cls_pred = [
                mlvl_cls_scores[lvl][img_id].view(-1, self.cls_out_channels)
                for lvl in range(num_levels)
            ]
            img_mask_feats = mask_feats[[img_id]]
            img_kernel_pred = [
                mlvl_kernel_preds[lvl][img_id].permute(1, 2, 0).view(
                    -1, self.kernel_out_channels) for lvl in range(num_levels)
            ]
            img_cls_pred = torch.cat(img_cls_pred, dim=0)
            img_kernel_pred = torch.cat(img_kernel_pred, dim=0)
            result = self._get_results_single(
                img_kernel_pred,
                img_cls_pred,
                img_mask_feats,
                img_meta=img_metas[img_id])
            result_list.append(result)
        return result_list

    def _get_results_single(self,
                            kernel_preds,
                            cls_scores,
                            mask_feats,
                            img_meta,
                            cfg=None):
        """获取单张图像上处理过的mask预测结果.

        Args:
            kernel_preds (Tensor): 所有层级所有位置上输出的动态卷积权重,
                [nl * num_grid**2, kernel_c]
            cls_scores (Tensor): 所有层级所有位置上的cls输出,[nl * num_grids**2, nc].
            mask_feats (Tensor): [1, mask_out_c, batch_h//4, batch_w//4].
            img_meta (dict): 当前图片元信息.
            cfg (dict, optional): 测试阶段使用的配置.默认: None.

        Returns:
            :obj:`InstanceData`: 单张图像的处理结果.一种储存分割结果的特殊数据结构
             通常包含以下键.
                - scores (Tensor): pred_cls_score, [num_instance,].
                - labels (Tensor): pred_cls_ind, [num_instance,].
                - masks (Tensor): pred_mask, [num_instances, H, W], 原始图像尺寸.
        """

        def empty_results(results, cls_scores):
            """Generate a empty results."""
            results.scores = cls_scores.new_ones(0)
            results.masks = cls_scores.new_zeros(0, *results.ori_shape[:2])
            results.labels = cls_scores.new_ones(0)
            return results

        cfg = self.test_cfg if cfg is None else cfg
        assert len(kernel_preds) == len(cls_scores)
        results = InstanceData(img_meta)

        featmap_size = mask_feats.size()[-2:]

        img_shape = results.img_shape
        ori_shape = results.ori_shape

        # overall info
        h, w, _ = img_shape
        upsampled_size = (featmap_size[0] * self.mask_stride,
                          featmap_size[1] * self.mask_stride)

        # process.
        score_mask = (cls_scores > cfg.score_thr)
        cls_scores = cls_scores[score_mask]
        if len(cls_scores) == 0:
            return empty_results(results, cls_scores)

        # cate_labels & kernel_preds
        inds = score_mask.nonzero()
        cls_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        lvl_interval = cls_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(lvl_interval[-1])

        strides[:lvl_interval[0]] *= self.strides[0]
        for lvl in range(1, self.num_levels):
            strides[lvl_interval[lvl -
                                 1]:lvl_interval[lvl]] *= self.strides[lvl]
        strides = strides[inds[:, 0]]

        # mask encoding.
        kernel_preds = kernel_preds.view(
            kernel_preds.size(0), -1, self.dynamic_conv_size,
            self.dynamic_conv_size)
        mask_preds = F.conv2d(
            mask_feats, kernel_preds, stride=1).squeeze(0).sigmoid()
        # mask.
        masks = mask_preds > cfg.mask_thr
        sum_masks = masks.sum((1, 2)).float()
        keep = sum_masks > strides
        if keep.sum() == 0:
            return empty_results(results, cls_scores)
        masks = masks[keep]
        mask_preds = mask_preds[keep]
        sum_masks = sum_masks[keep]
        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]

        # maskness.
        mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
        cls_scores *= mask_scores

        scores, labels, _, keep_inds = mask_matrix_nms(
            masks,
            cls_labels,
            cls_scores,
            mask_area=sum_masks,
            nms_pre=cfg.nms_pre,
            max_num=cfg.max_per_img,
            kernel=cfg.kernel,
            sigma=cfg.sigma,
            filter_thr=cfg.filter_thr)
        mask_preds = mask_preds[keep_inds]
        mask_preds = F.interpolate(
            mask_preds.unsqueeze(0),
            size=upsampled_size,
            mode='bilinear',
            align_corners=False)[:, :, :h, :w]
        mask_preds = F.interpolate(
            mask_preds,
            size=ori_shape[:2],
            mode='bilinear',
            align_corners=False).squeeze(0)
        masks = mask_preds > cfg.mask_thr

        results.masks = masks
        results.labels = labels
        results.scores = scores

        return results
