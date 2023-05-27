# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.core import InstanceData, mask_matrix_nms, multi_apply
from mmdet.core.utils import center_of_mass, generate_coordinate
from mmdet.models.builder import HEADS, build_loss
from .base_mask_head import BaseMaskHead


@HEADS.register_module()
class SOLOHead(BaseMaskHead):
    """SOLO mask head used in `SOLO: Segmenting Objects by Locations.

    <https://arxiv.org/abs/1912.04488>`_

    Args:
        num_classes (int): 背景类别数.
        in_channels (int): 输入特征维度.
        feat_channels (int): 隐藏层维度. 用于子类. 默认: 256.
        stacked_convs (int): head部分公共区域卷积数量. 默认: 4.
        strides (tuple): 各个特征图的下采样倍数.
        scale_ranges (tuple[tuple[int, int]]): 多层级mask的面积范围,((min1, max1), ) * nl.
        pos_scale (float): Constant scale factor to control the center region.
        num_grids (list[int]): Divided image into a uniform grids, each
            feature map has a different grid value. The number of output
            channels is grid ** 2. Default: [40, 36, 24, 16, 12].
        cls_down_index (int): 在stacked_convs中cls_down_index位置处的卷积操作之前
            将cls_feat进行下采样至num_grid[i]. 默认: 0.
        loss_mask (dict): mask loss配置.
        loss_cls (dict): cls loss配置.
        norm_cfg (dict): norm层配置.Default:
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        train_cfg (dict): head的训练配置.
        test_cfg (dict): head的测试配置.
        init_cfg (dict or list[dict], optional): head权重初始化配置.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        stacked_convs=4,
        strides=(4, 8, 16, 32, 64),
        scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
        pos_scale=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cls_down_index=0,
        loss_mask=None,
        loss_cls=None,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        train_cfg=None,
        test_cfg=None,
        init_cfg=[
            dict(type='Normal', layer='Conv2d', std=0.01),
            dict(
                type='Normal',
                std=0.01,
                bias_prob=0.01,
                override=dict(name='conv_mask_list')),
            dict(
                type='Normal',
                std=0.01,
                bias_prob=0.01,
                override=dict(name='conv_cls'))
        ],
    ):
        super(SOLOHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = self.num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.num_grids = num_grids
        # number of FPN feats
        self.num_levels = len(strides)
        assert self.num_levels == len(scale_ranges) == len(num_grids)
        self.scale_ranges = scale_ranges
        self.pos_scale = pos_scale

        self.cls_down_index = cls_down_index
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()

    def _init_layers(self):
        self.mask_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg))
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        self.conv_mask_list = nn.ModuleList()
        for num_grid in self.num_grids:
            self.conv_mask_list.append(
                nn.Conv2d(self.feat_channels, num_grid**2, 1))

        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

    def resize_feats(self, feats):
        """对第一层级特征进行下采样,最后一层特征进行上采样,其余层feat保持不变返回."""
        out = []
        for i in range(len(feats)):
            if i == 0:
                out.append(
                    F.interpolate(
                        feats[0],
                        size=feats[i + 1].shape[-2:],
                        mode='bilinear',
                        align_corners=False))
            elif i == len(feats) - 1:
                out.append(
                    F.interpolate(
                        feats[i],
                        size=feats[i - 1].shape[-2:],
                        mode='bilinear',
                        align_corners=False))
            else:
                out.append(feats[i])
        return out

    def forward(self, feats):
        assert len(feats) == self.num_levels
        feats = self.resize_feats(feats)
        mlvl_mask_preds = []
        mlvl_cls_preds = []
        for i in range(self.num_levels):
            x = feats[i]
            mask_feat = x
            cls_feat = x
            # 生成并cat特征坐标[-1, 1], [bs, c, h, w] -> [bs, c+2, h, w]
            coord_feat = generate_coordinate(mask_feat.size(),
                                             mask_feat.device)
            mask_feat = torch.cat([mask_feat, coord_feat], 1)

            for mask_layer in (self.mask_convs):
                mask_feat = mask_layer(mask_feat)

            mask_feat = F.interpolate(
                mask_feat, scale_factor=2, mode='bilinear')
            mask_pred = self.conv_mask_list[i](mask_feat)

            # cls branch
            for j, cls_layer in enumerate(self.cls_convs):
                if j == self.cls_down_index:
                    num_grid = self.num_grids[i]
                    cls_feat = F.interpolate(
                        cls_feat, size=num_grid, mode='bilinear')
                cls_feat = cls_layer(cls_feat)

            cls_pred = self.conv_cls(cls_feat)

            if not self.training:
                feat_wh = feats[0].size()[-2:]
                upsampled_size = (feat_wh[0] * 2, feat_wh[1] * 2)
                mask_pred = F.interpolate(
                    mask_pred.sigmoid(), size=upsampled_size, mode='bilinear')
                cls_pred = cls_pred.sigmoid()
                # get local maximum
                local_max = F.max_pool2d(cls_pred, 2, stride=1, padding=1)
                keep_mask = local_max[:, :, :-1, :-1] == cls_pred
                cls_pred = cls_pred * keep_mask

            mlvl_mask_preds.append(mask_pred)
            mlvl_cls_preds.append(cls_pred)
        return mlvl_mask_preds, mlvl_cls_preds

    def loss(self,
             mlvl_mask_preds,
             mlvl_cls_preds,
             gt_labels,
             gt_masks,
             img_metas,
             gt_bboxes=None,
             **kwargs):
        """计算batch幅图像的损失.

        Args:
            mlvl_mask_preds (list[Tensor]): 多层级的mask输出, [[bs, num_grids**2, h, w], ] * nl.
            mlvl_cls_preds (list[Tensor]): 多层级的cls输出. [[bs, nc, num_grid, num_grid], ] * nl.
                (batch_size, num_classes, num_grids ,num_grids).
            gt_labels (list[Tensor]): [[num_gt, ], ] * bs.
            gt_masks (list[Tensor]): [[num_gt, batch_h, batch_w], ] * bs.
            img_metas (list[dict]): [dict(), ] * bs.
            gt_bboxes (list[Tensor]): [[num_gt, 4], ] * bs. Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_levels = self.num_levels
        num_imgs = len(gt_labels)

        featmap_sizes = [featmap.size()[-2:] for featmap in mlvl_mask_preds]

        # pos_mask是bool类型的值,代表该位置是否是正样本区域.
        # [[[sum(pos_mask), h, w], * nl], ] * bs
        # [[[num_grid, num_grid], * nl], ] * bs
        # [[[num_grid**2, ], * nl], ] * bs
        pos_mask_targets, labels, pos_masks = multi_apply(
            self._get_targets_single,
            gt_bboxes,
            gt_labels,
            gt_masks,
            featmap_sizes=featmap_sizes)

        # [[?], * nl] * bs -> [[?], * img] * nl
        mlvl_pos_mask_targets = [[] for _ in range(num_levels)]
        mlvl_pos_mask_preds = [[] for _ in range(num_levels)]
        mlvl_pos_masks = [[] for _ in range(num_levels)]
        mlvl_labels = [[] for _ in range(num_levels)]
        for img_id in range(num_imgs):
            assert num_levels == len(pos_mask_targets[img_id])
            for lvl in range(num_levels):
                mlvl_pos_mask_targets[lvl].append(
                    pos_mask_targets[img_id][lvl])
                mlvl_pos_mask_preds[lvl].append(
                    mlvl_mask_preds[lvl][img_id, pos_masks[img_id][lvl], ...])
                mlvl_pos_masks[lvl].append(pos_masks[img_id][lvl])
                mlvl_labels[lvl].append(labels[img_id][lvl].flatten())

        # cat multiple image
        temp_mlvl_cls_preds = []
        for lvl in range(num_levels):
            # 将同一层级上的batch幅数据cat到一起,[bs*sum(pos_mask), h, w]
            mlvl_pos_mask_targets[lvl] = torch.cat(
                mlvl_pos_mask_targets[lvl], dim=0)
            mlvl_pos_mask_preds[lvl] = torch.cat(
                mlvl_pos_mask_preds[lvl], dim=0)
            # 同上,[bs * (num_grid**2), ]
            mlvl_pos_masks[lvl] = torch.cat(mlvl_pos_masks[lvl], dim=0)
            mlvl_labels[lvl] = torch.cat(mlvl_labels[lvl], dim=0)
            # [bs, num_grid ,num_grid, nc] -> [bs * (num_grid**2), nc]
            temp_mlvl_cls_preds.append(mlvl_cls_preds[lvl].permute(
                0, 2, 3, 1).reshape(-1, self.cls_out_channels))
        # batch幅图像上正样本区域bool值总和.
        num_pos = sum(item.sum() for item in mlvl_pos_masks)
        # dice loss
        loss_mask = []
        for pred, target in zip(mlvl_pos_mask_preds, mlvl_pos_mask_targets):
            if pred.size()[0] == 0:
                loss_mask.append(pred.sum().unsqueeze(0))
                continue
            loss_mask.append(
                self.loss_mask(pred, target, reduction_override='none'))
        if num_pos > 0:
            loss_mask = torch.cat(loss_mask).sum() / num_pos
        else:
            loss_mask = torch.cat(loss_mask).mean()

        flatten_labels = torch.cat(mlvl_labels)
        flatten_cls_preds = torch.cat(temp_mlvl_cls_preds)
        loss_cls = self.loss_cls(
            flatten_cls_preds, flatten_labels, avg_factor=num_pos + 1)
        return dict(loss_mask=loss_mask, loss_cls=loss_cls)

    def _get_targets_single(self,
                            gt_bboxes,
                            gt_labels,
                            gt_masks,
                            featmap_sizes=None):
        """计算单个图像的mask/cls_target.

        Args:
            gt_bboxes (Tensor): [num_gt, 4].
            gt_labels (Tensor): [num_gt, ].
            gt_masks (Tensor): [num_gt, batch_h, batch_w].
            featmap_sizes (list[:obj:`torch.size`]): [[h, w], ] * nl. 默认: None.

        Returns:
            - mlvl_pos_mask_targets (list[Tensor]):  [[num_pos, h, w], ] * nl.
                正样本的mask_target, 由gt_instance直接resize过来的.
            - mlvl_labels (list[Tensor]): [[num_grid, num_grid], ] * nl.
                gt box中心0.2倍宽高区域内值为gt_label.
            - mlvl_pos_masks (list[Tensor]): [[num_grid**2, ], ] * nl.
        """
        device = gt_labels.device
        gt_areas = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) *
                              (gt_bboxes[:, 3] - gt_bboxes[:, 1]))

        mlvl_pos_mask_targets = []
        mlvl_labels = []
        mlvl_pos_masks = []
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides,
                       featmap_sizes, self.num_grids):

            mask_target = torch.zeros(
                [num_grid**2, featmap_size[0], featmap_size[1]],
                dtype=torch.uint8,
                device=device)
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
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
                    mask_target.new_zeros(0, featmap_size[0], featmap_size[1]))
                mlvl_labels.append(labels)
                mlvl_pos_masks.append(pos_mask)
                continue
            hit_gt_bboxes = gt_bboxes[gt_inds]
            hit_gt_labels = gt_labels[gt_inds]
            hit_gt_masks = gt_masks[gt_inds, ...]

            # 以质心为中心生成gt_w/h*pos_scale(默认0.2)为宽高的正样本区域,
            # FCOS中,有类似的设计,它里面默认是整个gt范围都是正样本区域,
            # 不过它也有个center_samping参数来控制从gt中心向四周扩散范围
            # 如果开启,则范围固定为self.strides[lvl_idx] * radius(默认1.5)
            # SOLO里面正样本区域算是一种动态收紧,但感觉收的太紧了,仅为宽高的0.2倍
            # 这也是一个超参数,增大pos_scale则正样本数量增多,但是质量可能会下降一点.
            pos_w_ranges = 0.5 * (hit_gt_bboxes[:, 2] -
                                  hit_gt_bboxes[:, 0]) * self.pos_scale
            pos_h_ranges = 0.5 * (hit_gt_bboxes[:, 3] -
                                  hit_gt_bboxes[:, 1]) * self.pos_scale

            # Make sure hit_gt_masks has a value
            valid_mask_flags = hit_gt_masks.sum(dim=-1).sum(dim=-1) > 0
            # 此处/2是因为在Head部分的stack conv后又将所有层级的mask分支特征图放大了2倍
            output_stride = stride / 2

            for gt_mask, gt_label, pos_h_range, pos_w_range, \
                valid_mask_flag in \
                    zip(hit_gt_masks, hit_gt_labels, pos_h_ranges,
                        pos_w_ranges, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                # 这里应该是获取forward前的输入尺寸,但是这样固定写法感觉不太稳妥
                upsampled_size = (featmap_sizes[0][0] * 4,
                                  featmap_sizes[0][1] * 4)
                center_h, center_w = center_of_mass(gt_mask)

                # 获取质心坐标在当前层级num_grid尺寸下的坐标
                coord_w = int(
                    (center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int(
                    (center_h / upsampled_size[0]) // (1. / num_grid))

                # 并计算其在num_grid尺寸下的四个边界,left, top, right, down
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

                # 强制正样本区域在质心的外部
                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(left_box, coord_w - 1)
                right = min(right_box, coord_w + 1)

                # 在[num_grid, num_grid]大小的mask上的正样本区域赋予[0, nc-1]的label
                labels[top:(down + 1), left:(right + 1)] = gt_label
                # ins
                gt_mask = np.uint8(gt_mask.cpu().numpy())
                # Follow the original implementation, F.interpolate is
                # different from cv2 and opencv
                # 将gt mask放缩该层级上对应的stride/2倍.注意配置文件中的strides/2才是真实
                # 输入尺寸对应每个层级的下采样倍数.
                gt_mask = mmcv.imrescale(gt_mask, scale=1. / output_stride)
                gt_mask = torch.from_numpy(gt_mask).to(device=device)

                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        index = int(i * num_grid + j)
                        # [num_grid**2, featmap_size[0], featmap_size[1]]
                        # mask_target 与 gt_mask的后两个维度shape本质是有区别,
                        # gt_mask在rescale之前是pad_shape, -> pad_shape//4
                        # 而mask_target是batch_shape, -> batch_shape//4
                        # 在collect_fn中为了将batch幅数据shape保持一致,
                        # 会在pad_shape的基础上在右下角额外进行padding从而得到batch_shape.
                        mask_target[index, :gt_mask.shape[0], :gt_mask.
                                    shape[1]] = gt_mask
                        pos_mask[index] = True
            mlvl_pos_mask_targets.append(mask_target[pos_mask])
            mlvl_labels.append(labels)
            mlvl_pos_masks.append(pos_mask)
        return mlvl_pos_mask_targets, mlvl_labels, mlvl_pos_masks

    def get_results(self, mlvl_mask_preds, mlvl_cls_scores, img_metas,
                    **kwargs):
        """获取多张图片的分割结果.

        Args:
            mlvl_mask_preds (list[Tensor]): 多层级mask输出结果.
                [[bs, num_grids**2 ,h ,w], ] * nl.
            mlvl_cls_scores (list[Tensor]): 多层级cls输出结果.
                [[bs, nc, num_grids ,num_grids], ] * nl.
            img_metas (list[dict]): batch幅图像的元信息.

        Returns:
            list[:obj:`InstanceData`]: 处理后的batch幅图像分割结果,
            [img_result, ] * bs, img_result包含以下键值.

                - scores (Tensor): pred_cls_score, [num_instance, ].
                - labels (Tensor): pred_cls_ind, [num_instance, ].
                - masks (Tensor): pred_mask, [num_instances, h, w].
        """
        mlvl_cls_scores = [
            item.permute(0, 2, 3, 1) for item in mlvl_cls_scores
        ]
        assert len(mlvl_mask_preds) == len(mlvl_cls_scores)
        num_levels = len(mlvl_cls_scores)

        results_list = []
        for img_id in range(len(img_metas)):
            # [[num_grids**2, nc], ] * nl, 以下皆是获取单张图像的多层级输出再cat到一起
            cls_pred_list = [
                mlvl_cls_scores[lvl][img_id].view(-1, self.cls_out_channels)
                for lvl in range(num_levels)
            ]
            # [[num_grids**2 ,h ,w], ] * nl
            mask_pred_list = [
                mlvl_mask_preds[lvl][img_id] for lvl in range(num_levels)
            ]
            # [nl * num_grids**2, nc], [nl * num_grids**2 ,h ,w]
            cls_pred_list = torch.cat(cls_pred_list, dim=0)
            mask_pred_list = torch.cat(mask_pred_list, dim=0)

            results = self._get_results_single(
                cls_pred_list, mask_pred_list, img_meta=img_metas[img_id])
            results_list.append(results)

        return results_list

    def _get_results_single(self, cls_scores, mask_preds, img_meta, cfg=None):
        """获取单张图像上处理过的mask预测结果.

        Args:
            cls_scores (Tensor): [nl * num_grids**2, nc].
            mask_preds (Tensor): [nl * num_grids**2, h ,w].
            img_meta (dict): 当前图片元信息.
            cfg (dict, optional): 测试阶段使用的配置.默认: None.

        Returns:
            :obj:`InstanceData`: 单张图像的处理结果.一种储存分割结果的特殊数据结构
             它通常包含以下键.
                - scores (Tensor): pred_cls_score, [num_instance,].
                - labels (Tensor): pred_cls_ind, [num_instance,].
                - masks (Tensor): pred_mask, [num_instances, H, W], 原始图像尺寸.
        """

        def empty_results(results, cls_scores):
            """生成一个空的分割结果."""
            results.scores = cls_scores.new_ones(0)
            results.masks = cls_scores.new_zeros(0, *results.ori_shape[:2])
            results.labels = cls_scores.new_ones(0)
            return results

        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(mask_preds)
        results = InstanceData(img_meta)

        featmap_size = mask_preds.size()[-2:]

        img_shape = results.img_shape
        ori_shape = results.ori_shape

        h, w, _ = img_shape
        upsampled_size = (featmap_size[0] * 4, featmap_size[1] * 4)

        # 1.根据score_thr阈值过滤掉cls_score低于该值的mask
        score_mask = (cls_scores > cfg.score_thr)
        cls_scores = cls_scores[score_mask]
        if len(cls_scores) == 0:
            return empty_results(results, cls_scores)

        # [?, 2]类似形状的数据,其中"2"维度上的第二列代表nc上的索引也即label
        inds = score_mask.nonzero()
        cls_labels = inds[:, 1]

        # [1600., 2896., 3472., 3728., 3872.]
        lvl_interval = cls_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = cls_scores.new_ones(lvl_interval[-1])
        strides[:lvl_interval[0]] *= self.strides[0]
        for lvl in range(1, self.num_levels):
            # 将strides上第lvl+1层级的值乘以相应层级下采样倍数
            strides[lvl_interval[lvl-1]:lvl_interval[lvl]] *= self.strides[lvl]
        strides = strides[inds[:, 0]]
        mask_preds = mask_preds[inds[:, 0]]

        # 2.根据mask_thr过滤掉低于该值的mask, 并计算出满足条件的mask总数
        masks = mask_preds > cfg.mask_thr
        sum_masks = masks.sum((1, 2)).float()
        # 3.过滤掉任意一个在[h, w]上面积区域小于strides值的实例
        keep = sum_masks > strides
        if keep.sum() == 0:
            return empty_results(results, cls_scores)
        masks = masks[keep]
        mask_preds = mask_preds[keep]
        sum_masks = sum_masks[keep]
        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]

        # maskness. 计算num_grid**2上任意一个mask区域的平均得分,计算方式就是统计
        # mask在[h, w]区域上的满足条件的score总和再除以区域面积
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
        # upsampled_size是指batch_shape(在右下角进行padding),
        # 而img_shape是指ori_img进行Resize后的尺寸,ori_shape才是图像原始尺寸
        mask_preds = F.interpolate(
            mask_preds.unsqueeze(0), size=upsampled_size,
            mode='bilinear')[:, :, :h, :w]
        mask_preds = F.interpolate(
            mask_preds, size=ori_shape[:2], mode='bilinear').squeeze(0)
        masks = mask_preds > cfg.mask_thr

        results.masks = masks
        results.labels = labels
        results.scores = scores

        return results


@HEADS.register_module()
class DecoupledSOLOHead(SOLOHead):
    """Decoupled SOLO mask head used in `SOLO: Segmenting Objects by Locations.

    <https://arxiv.org/abs/1912.04488>`_

    Args:
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 *args,
                 init_cfg=[
                     dict(type='Normal', layer='Conv2d', std=0.01),
                     dict(
                         type='Normal',
                         std=0.01,
                         bias_prob=0.01,
                         override=dict(name='conv_mask_list_x')),
                     dict(
                         type='Normal',
                         std=0.01,
                         bias_prob=0.01,
                         override=dict(name='conv_mask_list_y')),
                     dict(
                         type='Normal',
                         std=0.01,
                         bias_prob=0.01,
                         override=dict(name='conv_cls'))
                 ],
                 **kwargs):
        super(DecoupledSOLOHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        self.mask_convs_x = nn.ModuleList()
        self.mask_convs_y = nn.ModuleList()
        self.cls_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels + 1 if i == 0 else self.feat_channels
            self.mask_convs_x.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg))
            self.mask_convs_y.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg))

            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg))

        self.conv_mask_list_x = nn.ModuleList()
        self.conv_mask_list_y = nn.ModuleList()
        for num_grid in self.num_grids:
            self.conv_mask_list_x.append(
                nn.Conv2d(self.feat_channels, num_grid, 3, padding=1))
            self.conv_mask_list_y.append(
                nn.Conv2d(self.feat_channels, num_grid, 3, padding=1))
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

    def forward(self, feats):
        assert len(feats) == self.num_levels
        feats = self.resize_feats(feats)
        mask_preds_x = []
        mask_preds_y = []
        cls_preds = []
        for i in range(self.num_levels):
            x = feats[i]
            mask_feat = x
            cls_feat = x
            # generate and concat the coordinate
            coord_feat = generate_coordinate(mask_feat.size(),
                                             mask_feat.device)
            mask_feat_x = torch.cat([mask_feat, coord_feat[:, 0:1, ...]], 1)
            mask_feat_y = torch.cat([mask_feat, coord_feat[:, 1:2, ...]], 1)

            for mask_layer_x, mask_layer_y in \
                    zip(self.mask_convs_x, self.mask_convs_y):
                mask_feat_x = mask_layer_x(mask_feat_x)
                mask_feat_y = mask_layer_y(mask_feat_y)

            mask_feat_x = F.interpolate(
                mask_feat_x, scale_factor=2, mode='bilinear')
            mask_feat_y = F.interpolate(
                mask_feat_y, scale_factor=2, mode='bilinear')

            mask_pred_x = self.conv_mask_list_x[i](mask_feat_x)
            mask_pred_y = self.conv_mask_list_y[i](mask_feat_y)

            # cls branch
            for j, cls_layer in enumerate(self.cls_convs):
                if j == self.cls_down_index:
                    num_grid = self.num_grids[i]
                    cls_feat = F.interpolate(
                        cls_feat, size=num_grid, mode='bilinear')
                cls_feat = cls_layer(cls_feat)

            cls_pred = self.conv_cls(cls_feat)

            if not self.training:
                feat_wh = feats[0].size()[-2:]
                upsampled_size = (feat_wh[0] * 2, feat_wh[1] * 2)
                mask_pred_x = F.interpolate(
                    mask_pred_x.sigmoid(),
                    size=upsampled_size,
                    mode='bilinear')
                mask_pred_y = F.interpolate(
                    mask_pred_y.sigmoid(),
                    size=upsampled_size,
                    mode='bilinear')
                cls_pred = cls_pred.sigmoid()
                # get local maximum
                local_max = F.max_pool2d(cls_pred, 2, stride=1, padding=1)
                keep_mask = local_max[:, :, :-1, :-1] == cls_pred
                cls_pred = cls_pred * keep_mask

            mask_preds_x.append(mask_pred_x)
            mask_preds_y.append(mask_pred_y)
            cls_preds.append(cls_pred)
        return mask_preds_x, mask_preds_y, cls_preds

    def loss(self,
             mlvl_mask_preds_x,
             mlvl_mask_preds_y,
             mlvl_cls_preds,
             gt_labels,
             gt_masks,
             img_metas,
             gt_bboxes=None,
             **kwargs):
        """计算batch幅图像的损失.

        Args:
            mlvl_mask_preds_x (list[Tensor]): 多层级x分支的mask输出, [[bs, num_grids, h ,w], ] * nl.
            mlvl_mask_preds_y (list[Tensor]): 多层级y分支的mask输出, [[bs, num_grids, h, w], ] * nl.
            mlvl_cls_preds (list[Tensor]): 多层级的cls输出. [[bs, nc ,num_grid, num_grids], ] * nl.
            gt_labels (list[Tensor]): [[num_gt, ], ] * bs.
            gt_masks (list[Tensor]): [[num_gt, batch_h, batch_w], ] * bs.
            img_metas (list[dict]): [dict(), ] * bs.
            gt_bboxes (list[Tensor]): [[num_gt, 4], ] * bs. Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_levels = self.num_levels
        num_imgs = len(gt_labels)
        featmap_sizes = [featmap.size()[-2:] for featmap in mlvl_mask_preds_x]

        # [[[[num_pos, h, w], ] * nl], ] * bs, [[[num_grid, num_grid], ] * nl, ] * bs
        # [[[num_grid, 2], ] * nl, ] * bs
        pos_mask_targets, labels, \
            xy_pos_indexes = \
            multi_apply(self._get_targets_single,
                        gt_bboxes,
                        gt_labels,
                        gt_masks,
                        featmap_sizes=featmap_sizes)

        # change from the outside list meaning multi images
        # to the outside list meaning multi levels
        mlvl_pos_mask_targets = [[] for _ in range(num_levels)]
        mlvl_pos_mask_preds_x = [[] for _ in range(num_levels)]
        mlvl_pos_mask_preds_y = [[] for _ in range(num_levels)]
        mlvl_labels = [[] for _ in range(num_levels)]
        for img_id in range(num_imgs):

            for lvl in range(num_levels):
                mlvl_pos_mask_targets[lvl].append(
                    pos_mask_targets[img_id][lvl])
                mlvl_pos_mask_preds_x[lvl].append(
                    mlvl_mask_preds_x[lvl][img_id,
                                           xy_pos_indexes[img_id][lvl][:, 1]])
                mlvl_pos_mask_preds_y[lvl].append(
                    mlvl_mask_preds_y[lvl][img_id,
                                           xy_pos_indexes[img_id][lvl][:, 0]])
                mlvl_labels[lvl].append(labels[img_id][lvl].flatten())

        # cat multiple image
        temp_mlvl_cls_preds = []
        for lvl in range(num_levels):
            mlvl_pos_mask_targets[lvl] = torch.cat(
                mlvl_pos_mask_targets[lvl], dim=0)
            mlvl_pos_mask_preds_x[lvl] = torch.cat(
                mlvl_pos_mask_preds_x[lvl], dim=0)
            mlvl_pos_mask_preds_y[lvl] = torch.cat(
                mlvl_pos_mask_preds_y[lvl], dim=0)
            mlvl_labels[lvl] = torch.cat(mlvl_labels[lvl], dim=0)
            temp_mlvl_cls_preds.append(mlvl_cls_preds[lvl].permute(
                0, 2, 3, 1).reshape(-1, self.cls_out_channels))

        num_pos = 0.
        # dice loss
        loss_mask = []
        for pred_x, pred_y, target in \
                zip(mlvl_pos_mask_preds_x,
                    mlvl_pos_mask_preds_y, mlvl_pos_mask_targets):
            num_masks = pred_x.size(0)
            if num_masks == 0:
                # make sure can get grad
                loss_mask.append((pred_x.sum() + pred_y.sum()).unsqueeze(0))
                continue
            num_pos += num_masks
            pred_mask = pred_y.sigmoid() * pred_x.sigmoid()
            loss_mask.append(
                self.loss_mask(pred_mask, target, reduction_override='none'))
        if num_pos > 0:
            loss_mask = torch.cat(loss_mask).sum() / num_pos
        else:
            loss_mask = torch.cat(loss_mask).mean()

        # cate
        flatten_labels = torch.cat(mlvl_labels)
        flatten_cls_preds = torch.cat(temp_mlvl_cls_preds)

        loss_cls = self.loss_cls(
            flatten_cls_preds, flatten_labels, avg_factor=num_pos + 1)
        return dict(loss_mask=loss_mask, loss_cls=loss_cls)

    def _get_targets_single(self,
                            gt_bboxes,
                            gt_labels,
                            gt_masks,
                            featmap_sizes=None):
        """计算单个图像的mask/cls_target.

        Args:
            gt_bboxes (Tensor): [num_gt, 4].
            gt_labels (Tensor): [num_gt, ].
            gt_masks (Tensor): [num_gt, batch_h, batch_w].
            featmap_sizes (list[:obj:`torch.size`]): [[h, w], ] * nl. 默认: None.

        Returns:
            Tuple: Usually returns a tuple containing targets for predictions.

                - mlvl_pos_mask_targets (list[Tensor]): [[num_pos, h, w], ] * nl.
                    正样本的mask_target, 由gt_instance直接resize过来的.
                - mlvl_labels (list[Tensor]): [[num_grid, num_grid], ] * nl.
                    gt box中心0.2倍宽高区域内值为gt_label.
                - mlvl_xy_pos_indexes (list[Tensor]): 优化了SOLOv1中的实例位置表示方法,
                  由[[num_grid, 2], ] * nl, 其中2代表x与y索引.
        """
        # [[num_pos, h, w], ] * nl, [[num_grid, num_grid], ] * nl, [[num_grid**2, ], ] * nl.
        mlvl_pos_mask_targets, mlvl_labels, \
            mlvl_pos_masks = \
            super()._get_targets_single(gt_bboxes, gt_labels, gt_masks,
                                        featmap_sizes=featmap_sizes)

        # 注意.nonzero表示的是二维坐标,所以[num_pos, 2]第一列是纵向y坐标,第二列是横向x坐标.
        mlvl_xy_pos_indexes = [(item - self.num_classes).nonzero()
                               for item in mlvl_labels]

        return mlvl_pos_mask_targets, mlvl_labels, mlvl_xy_pos_indexes

    def get_results(self,
                    mlvl_mask_preds_x,
                    mlvl_mask_preds_y,
                    mlvl_cls_scores,
                    img_metas,
                    rescale=None,
                    **kwargs):
        """获取多张图片的分割结果.

        Args:
            mlvl_mask_preds_x (list[Tensor]): 多层级x分支的mask输出结果.
                [[bs, num_grids ,h ,w], ] * nl.
            mlvl_mask_preds_y (list[Tensor]): 多层级y分支的mask输出结果.
                [[bs, num_grids ,h ,w], ] * nl.
            mlvl_cls_scores (list[Tensor]): 多层级cls输出结果.
                [[bs, nc, num_grids ,num_grids], ] * nl.
            img_metas (list[dict]): batch幅图像的元信息.

        Returns:
            list[:obj:`InstanceData`]: 处理后的batch幅图像分割结果,
            [img_result, ] * bs, img_result包含以下键值.

                - scores (Tensor): pred_cls_score, [num_instance,].
                - labels (Tensor): pred_cls_ind, [num_instance,].
                - masks (Tensor): pred_mask, [num_instances, h, w].
        """
        mlvl_cls_scores = [
            item.permute(0, 2, 3, 1) for item in mlvl_cls_scores
        ]
        assert len(mlvl_mask_preds_x) == len(mlvl_cls_scores)
        num_levels = len(mlvl_cls_scores)

        results_list = []
        for img_id in range(len(img_metas)):
            cls_pred_list = [
                mlvl_cls_scores[i][img_id].view(
                    -1, self.cls_out_channels).detach()
                for i in range(num_levels)
            ]
            mask_pred_list_x = [
                mlvl_mask_preds_x[i][img_id] for i in range(num_levels)
            ]
            mask_pred_list_y = [
                mlvl_mask_preds_y[i][img_id] for i in range(num_levels)
            ]

            cls_pred_list = torch.cat(cls_pred_list, dim=0)
            mask_pred_list_x = torch.cat(mask_pred_list_x, dim=0)
            mask_pred_list_y = torch.cat(mask_pred_list_y, dim=0)

            results = self._get_results_single(
                cls_pred_list,
                mask_pred_list_x,
                mask_pred_list_y,
                img_meta=img_metas[img_id],
                cfg=self.test_cfg)
            results_list.append(results)
        return results_list

    def _get_results_single(self, cls_scores, mask_preds_x, mask_preds_y,
                            img_meta, cfg):
        """获取单张图像上处理过的mask预测结果.

        Args:
            cls_scores (Tensor): [nl * num_grids**2, nc].
            mask_preds_x (Tensor): [nl * num_grids, h ,w].
            mask_preds_y (Tensor): [nl * num_grids, h ,w].
            img_meta (dict): 当前图片元信息.
            cfg (dict): 测试阶段使用的配置.默认: None.

        Returns:
            :obj:`InstanceData`: 处理后的单幅图像分割结果.
             它通常包含以下键.

                - scores (Tensor): pred_cls_score, [num_instance,].
                - labels (Tensor): pred_cls_ind, [num_instance,].
                - masks (Tensor): pred_mask, [num_instances, H, W], 原始图像尺寸.
        """

        def empty_results(results, cls_scores):
            """生成一个空的分割结果."""
            results.scores = cls_scores.new_ones(0)
            results.masks = cls_scores.new_zeros(0, *results.ori_shape[:2])
            results.labels = cls_scores.new_ones(0)
            return results

        cfg = self.test_cfg if cfg is None else cfg

        # 获取图像的形状信息, 包括原始形状和缩放后形状
        results = InstanceData(img_meta)
        img_shape = results.img_shape
        ori_shape = results.ori_shape
        h, w, _ = img_shape

        # 获取特征图尺寸和上采样后的尺寸
        featmap_size = mask_preds_x.size()[-2:]
        upsampled_size = (featmap_size[0] * 4, featmap_size[1] * 4)

        # 根据分类得分 cls_scores 找到所有置信度大于阈值 cfg.score_thr 的部分
        score_mask = (cls_scores > cfg.score_thr)
        cls_scores = cls_scores[score_mask]
        # [?, 2] 第一列表示满足条件的位置索引∈[0, nl*num_grid**2), 第二列表示类别索引∈[0, nc)
        inds = score_mask.nonzero()

        # num_grids = [40, 36, 24, 16, 12] 注意这里的操作是为了后续分割cat到一起的多层级数据.
        # [1600., 2896., 3472., 3728., 3872.], 3872
        lvl_interval = inds.new_tensor(self.num_grids).pow(2).cumsum(0)  # num_grid**2前缀和
        num_all_points = lvl_interval[-1]

        # 定义相关变量, 除了seg_size为[nl, ]. 其余皆是[nl*num_grid**2, ]
        lvl_start_index = inds.new_ones(num_all_points)       # 实例所属层级的num_grid**2前缀和
        num_grids = inds.new_ones(num_all_points)             # 实例所属层级的num_grid值
        seg_size = inds.new_tensor(self.num_grids).cumsum(0)  # num_grid前缀和
        mask_lvl_start_index = inds.new_ones(num_all_points)  # 实例所属层级的num_grid前缀和
        strides = inds.new_ones(num_all_points)               # 实例所属层级的stride值

        # 初始化相关变量在第一层级上的值,方便下面循环中被后续层级引用
        lvl_start_index[:lvl_interval[0]] *= 0
        mask_lvl_start_index[:lvl_interval[0]] *= 0
        num_grids[:lvl_interval[0]] *= self.num_grids[0]
        strides[:lvl_interval[0]] *= self.strides[0]

        for lvl in range(1, self.num_levels):
            # 将第lvl+1层级上对应的num_grid**2个值乘以(前lvl层num_grid**2总和)
            lvl_start_index[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= \
                lvl_interval[lvl - 1]
            # 将第lvl+1层级上对应的num_grid**2个值乘以(前lvl层num_grid总和)
            mask_lvl_start_index[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= \
                seg_size[lvl - 1]
            # 将num_grids第lvl+1层级上对应的num_grid**2个值乘以(第lvl+1层num_grid值(int))
            num_grids[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= \
                self.num_grids[lvl]
            # 将strides第lvl+1层级上对应的num_grid**2个值乘以(第lvl+1层strides值(int))
            strides[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= \
                self.strides[lvl]

        # 1.根据score_thr阈值筛选出cls_score高于该值的有效索引
        lvl_start_index = lvl_start_index[inds[:, 0]]
        mask_lvl_start_index = mask_lvl_start_index[inds[:, 0]]
        num_grids = num_grids[inds[:, 0]]
        strides = strides[inds[:, 0]]

        # inds[:, 0]是nl*(num_grid**2)上的索引,lvl_start_index则是不同层级上累加的num_grid**2前缀和,
        # 减法之后就得到满足score_thr阈值条件的实例在对应层级num_grid**2上的索引
        # 再做整除和取余则就得到纵向y坐标和横向x坐标了.
        y_lvl_offset = (inds[:, 0] - lvl_start_index) // num_grids
        x_lvl_offset = (inds[:, 0] - lvl_start_index) % num_grids
        # mask_lvl_start_index表示满足score_thr阈值条件的实例在对应层级的num_grid前缀和
        # 再加上num_grid的纵向/横向索引则就对齐了网络输出的mask_x/y,[nl*num_grid, h, w]
        y_inds = mask_lvl_start_index + y_lvl_offset
        x_inds = mask_lvl_start_index + x_lvl_offset

        cls_labels = inds[:, 1]
        # 令mask_preds表示为mask_x*mask_y
        mask_preds = mask_preds_x[x_inds, ...] * mask_preds_y[y_inds, ...]

        # 2.根据mask_thr过滤掉低于该值的mask, 并计算出满足条件的mask总数
        masks = mask_preds > cfg.mask_thr
        sum_masks = masks.sum((1, 2)).float()
        # 3.过滤掉任意一个在[h, w]上面积区域小于strides值的实例
        keep = sum_masks > strides
        if keep.sum() == 0:
            return empty_results(results, cls_scores)

        masks = masks[keep]
        mask_preds = mask_preds[keep]
        sum_masks = sum_masks[keep]
        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]

        # maskness. 计算num_grid**2上任意一个mask区域的平均得分,计算方式就是统计
        # mask在[h, w]区域上的满足条件的score总和再除以区域面积
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
        # upsampled_size是指batch_shape(在右下角进行padding),
        # 而img_shape是指ori_img进行Resize后的尺寸,ori_shape才是图像原始尺寸
        mask_preds = F.interpolate(
            mask_preds.unsqueeze(0), size=upsampled_size,
            mode='bilinear')[:, :, :h, :w]
        mask_preds = F.interpolate(
            mask_preds, size=ori_shape[:2], mode='bilinear').squeeze(0)
        masks = mask_preds > cfg.mask_thr

        results.masks = masks
        results.labels = labels
        results.scores = scores

        return results


@HEADS.register_module()
class DecoupledSOLOLightHead(DecoupledSOLOHead):
    """Decoupled Light SOLO mask head used in `SOLO: Segmenting Objects by
    Locations <https://arxiv.org/abs/1912.04488>`_
    前半部分是SOLOHead,后面上采样后分叉出x/y分支保持与DecoupledSOLOHead一致的输出.

    Args:
        with_dcn (bool): Whether use dcn in mask_convs and cls_convs,
            default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 *args,
                 dcn_cfg=None,
                 init_cfg=[
                     dict(type='Normal', layer='Conv2d', std=0.01),
                     dict(
                         type='Normal',
                         std=0.01,
                         bias_prob=0.01,
                         override=dict(name='conv_mask_list_x')),
                     dict(
                         type='Normal',
                         std=0.01,
                         bias_prob=0.01,
                         override=dict(name='conv_mask_list_y')),
                     dict(
                         type='Normal',
                         std=0.01,
                         bias_prob=0.01,
                         override=dict(name='conv_cls'))
                 ],
                 **kwargs):
        assert dcn_cfg is None or isinstance(dcn_cfg, dict)
        self.dcn_cfg = dcn_cfg
        super(DecoupledSOLOLightHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        self.mask_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            if self.dcn_cfg is not None\
                    and i == self.stacked_convs - 1:
                conv_cfg = self.dcn_cfg
            else:
                conv_cfg = None

            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))

            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.conv_mask_list_x = nn.ModuleList()
        self.conv_mask_list_y = nn.ModuleList()
        for num_grid in self.num_grids:
            self.conv_mask_list_x.append(
                nn.Conv2d(self.feat_channels, num_grid, 3, padding=1))
            self.conv_mask_list_y.append(
                nn.Conv2d(self.feat_channels, num_grid, 3, padding=1))
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

    def forward(self, feats):
        assert len(feats) == self.num_levels
        feats = self.resize_feats(feats)
        mask_preds_x = []
        mask_preds_y = []
        cls_preds = []
        for i in range(self.num_levels):
            x = feats[i]
            mask_feat = x
            cls_feat = x
            # generate and concat the coordinate
            coord_feat = generate_coordinate(mask_feat.size(),
                                             mask_feat.device)
            mask_feat = torch.cat([mask_feat, coord_feat], 1)

            for mask_layer in self.mask_convs:
                mask_feat = mask_layer(mask_feat)

            mask_feat = F.interpolate(
                mask_feat, scale_factor=2, mode='bilinear')

            mask_pred_x = self.conv_mask_list_x[i](mask_feat)
            mask_pred_y = self.conv_mask_list_y[i](mask_feat)

            # cls branch
            for j, cls_layer in enumerate(self.cls_convs):
                if j == self.cls_down_index:
                    num_grid = self.num_grids[i]
                    cls_feat = F.interpolate(
                        cls_feat, size=num_grid, mode='bilinear')
                cls_feat = cls_layer(cls_feat)

            cls_pred = self.conv_cls(cls_feat)

            if not self.training:
                feat_wh = feats[0].size()[-2:]
                upsampled_size = (feat_wh[0] * 2, feat_wh[1] * 2)
                mask_pred_x = F.interpolate(
                    mask_pred_x.sigmoid(),
                    size=upsampled_size,
                    mode='bilinear')
                mask_pred_y = F.interpolate(
                    mask_pred_y.sigmoid(),
                    size=upsampled_size,
                    mode='bilinear')
                cls_pred = cls_pred.sigmoid()
                # get local maximum
                local_max = F.max_pool2d(cls_pred, 2, stride=1, padding=1)
                keep_mask = local_max[:, :, :-1, :-1] == cls_pred
                cls_pred = cls_pred * keep_mask

            mask_preds_x.append(mask_pred_x)
            mask_preds_y.append(mask_pred_y)
            cls_preds.append(cls_pred)
        return mask_preds_x, mask_preds_y, cls_preds
