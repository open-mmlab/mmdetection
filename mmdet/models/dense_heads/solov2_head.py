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
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels of the mask feature
             map branch.
        start_level (int): The starting feature map level from RPN that
             will be used to predict the mask feature map.
        end_level (int): The ending feature map level from rpn that
             will be used to predict the mask feature map.
        out_channels (int): Number of output channels of the mask feature
             map branch. This is the channel count of the mask
             feature map that to be dynamically convolved with the predicted
             kernel.
        mask_stride (int): Downsample factor of the mask feature map output.
            Default: 4.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
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

            feature_add_all_level += self.convs_all_levels[i](input_p)

        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred


@HEADS.register_module()
class SOLOV2Head(SOLOHead):
    """SOLOv2 mask head used in `SOLOv2: Dynamic and Fast Instance
    Segmentation. <https://arxiv.org/pdf/2003.10152>`_

    Args:
        mask_feature_head (dict): Config of SOLOv2MaskFeatHead.
        dynamic_conv_size (int): Dynamic Conv kernel size. Default: 1.
        dcn_cfg (dict): Dcn conv configurations in kernel_convs and cls_conv.
            default: None.
        dcn_apply_to_all_conv (bool): Whether to use dcn in every layer of
            kernel_convs and cls_convs, or only the last layer. It shall be set
            `True` for the normal version of SOLOv2 and `False` for the
            light-weight version. default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
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

        # update the in_channels of mask_feature_head
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
        """Compute targets for predictions of single image.

        Args:
            gt_bboxes (Tensor): Ground truth bbox of each instance,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth label of each instance,
                shape (num_gts,).
            gt_masks (Tensor): Ground truth mask of each instance,
                shape (num_gts, h, w).
            featmap_sizes (:obj:`torch.size`): Size of UNified mask
                feature map used to generate instance segmentation
                masks by dynamic convolution, each element means
                (feat_h, feat_w). Default: None.

        Returns:
            Tuple: Usually returns a tuple containing targets for predictions.

                - mlvl_pos_mask_targets (list[Tensor]): Each element represent
                  the binary mask targets for positive points in this
                  level, has shape (num_pos, out_h, out_w).
                - mlvl_labels (list[Tensor]): Each element is
                  classification labels for all
                  points in this level, has shape
                  (num_grid, num_grid).
                - mlvl_pos_masks  (list[Tensor]): Each element is
                  a `BoolTensor` to represent whether the
                  corresponding point in single level
                  is positive, has shape (num_grid **2).
                - mlvl_pos_indexes  (list[list]): Each element
                  in the list contains the positive index in
                  corresponding level, has shape (num_pos).
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
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
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
            mlvl_kernel_preds (list[Tensor]): Multi-level dynamic kernel
                prediction. The kernel is used to generate instance
                segmentation masks by dynamic convolution. Each element in the
                list has shape
                (batch_size, kernel_out_channels, num_grids, num_grids).
            mlvl_cls_preds (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes, num_grids, num_grids).
            mask_feats (Tensor): Unified mask feature map used to generate
                instance segmentation masks by dynamic convolution. Has shape
                (batch_size, mask_out_channels, h, w).
            gt_labels (list[Tensor]): Labels of multiple images.
            gt_masks (list[Tensor]): Ground truth masks of multiple images.
                Each has shape (num_instances, h, w).
            img_metas (list[dict]): Meta information of multiple images.
            gt_bboxes (list[Tensor]): Ground truth bboxes of multiple
                images. Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_size = mask_feats.size()[-2:]

        pos_mask_targets, labels, pos_masks, pos_indexes = multi_apply(
            self._get_targets_single,
            gt_bboxes,
            gt_labels,
            gt_masks,
            featmap_size=featmap_size)

        mlvl_mask_targets = [
            torch.cat(lvl_mask_targets, 0)
            for lvl_mask_targets in zip(*pos_mask_targets)
        ]

        mlvl_pos_kernel_preds = []
        for lvl_kernel_preds, lvl_pos_indexes in zip(mlvl_kernel_preds,
                                                     zip(*pos_indexes)):
            lvl_pos_kernel_preds = []
            for img_lvl_kernel_preds, img_lvl_pos_indexes in zip(
                    lvl_kernel_preds, lvl_pos_indexes):
                img_lvl_pos_kernel_preds = img_lvl_kernel_preds.view(
                    img_lvl_kernel_preds.shape[0], -1)[:, img_lvl_pos_indexes]
                lvl_pos_kernel_preds.append(img_lvl_pos_kernel_preds)
            mlvl_pos_kernel_preds.append(lvl_pos_kernel_preds)

        # make multilevel mlvl_mask_pred
        mlvl_mask_preds = []
        for lvl_pos_kernel_preds in mlvl_pos_kernel_preds:
            lvl_mask_preds = []
            for img_id, img_lvl_pos_kernel_pred in enumerate(
                    lvl_pos_kernel_preds):
                if img_lvl_pos_kernel_pred.size()[-1] == 0:
                    continue
                img_mask_feats = mask_feats[[img_id]]
                h, w = img_mask_feats.shape[-2:]
                num_kernel = img_lvl_pos_kernel_pred.shape[1]
                img_lvl_mask_pred = F.conv2d(
                    img_mask_feats,
                    img_lvl_pos_kernel_pred.permute(1, 0).view(
                        num_kernel, -1, self.dynamic_conv_size,
                        self.dynamic_conv_size),
                    stride=1).view(-1, h, w)
                lvl_mask_preds.append(img_lvl_mask_pred)
            if len(lvl_mask_preds) == 0:
                lvl_mask_preds = None
            else:
                lvl_mask_preds = torch.cat(lvl_mask_preds, 0)
            mlvl_mask_preds.append(lvl_mask_preds)
        # dice loss
        num_pos = 0
        for img_pos_masks in pos_masks:
            for lvl_img_pos_masks in img_pos_masks:
                num_pos += lvl_img_pos_masks.count_nonzero()

        loss_mask = []
        for lvl_mask_preds, lvl_mask_targets in zip(mlvl_mask_preds,
                                                    mlvl_mask_targets):
            if lvl_mask_preds is None:
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

        # cate
        flatten_labels = [
            torch.cat(
                [img_lvl_labels.flatten() for img_lvl_labels in lvl_labels])
            for lvl_labels in zip(*labels)
        ]
        flatten_labels = torch.cat(flatten_labels)

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
            mlvl_kernel_preds (list[Tensor]): Multi-level dynamic kernel
                prediction. The kernel is used to generate instance
                segmentation masks by dynamic convolution. Each element in the
                list has shape
                (batch_size, kernel_out_channels, num_grids, num_grids).
            mlvl_cls_scores (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes, num_grids, num_grids).
            mask_feats (Tensor): Unified mask feature map used to generate
                instance segmentation masks by dynamic convolution. Has shape
                (batch_size, mask_out_channels, h, w).
            img_metas (list[dict]): Meta information of all images.

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
        """Get processed mask related results of single image.

        Args:
            kernel_preds (Tensor): Dynamic kernel prediction of all points
                in single image, has shape
                (num_points, kernel_out_channels).
            cls_scores (Tensor): Classification score of all points
                in single image, has shape (num_points, num_classes).
            mask_preds (Tensor): Mask prediction of all points in
                single image, has shape (num_points, feat_h, feat_w).
            img_meta (dict): Meta information of corresponding image.
            cfg (dict, optional): Config used in test phase.
                Default: None.

        Returns:
            :obj:`InstanceData`: Processed results of single image.
             it usually contains following keys.
                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
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
