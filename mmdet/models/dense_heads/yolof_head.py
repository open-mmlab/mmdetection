# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, bias_init_with_prob, constant_init, is_norm,
                      normal_init)
from mmcv.runner import force_fp32

from mmdet.core import anchor_inside_flags, multi_apply, reduce_mean, unmap
from ..builder import HEADS
from .anchor_head import AnchorHead

INF = 1e8


def levels_to_images(mlvl_tensor):
    """将一张图像的多层级特征图cat到一起.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    将 mlvl_tensor 中每个张量的形状从 (B, C, H, W) 转换为 (B, H*W , C), 再转换为
    [(H*W, C),] * B, 并将同一图像中所有级别上的特征图cat到一起.

    Args:
        mlvl_tensor (list[torch.Tensor]): [(B, C, H, W),] * num_level

    Returns:
        list[torch.Tensor]: [(num_level * H * W, C),] * B
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:  # [[bs, na * nc/4, h, w],] * num_level
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


@HEADS.register_module()
class YOLOFHead(AnchorHead):
    """YOLOFHead Paper link: https://arxiv.org/abs/2103.09460.

    Args:
        num_classes (int): 检测类别数量
        in_channels (List[int]): 每个scale的输入通道数量.
        cls_num_convs (int): cls分支的卷积数量.默认为2.
        reg_num_convs (int): reg分支的卷积数量.默认为4.
        norm_cfg (dict): 构造和配置norm层的字典.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_cls_convs=2,
                 num_reg_convs=4,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 **kwargs):
        self.num_cls_convs = num_cls_convs
        self.num_reg_convs = num_reg_convs
        self.norm_cfg = norm_cfg
        super(YOLOFHead, self).__init__(num_classes, in_channels, **kwargs)

    def _init_layers(self):
        cls_subnet = []
        bbox_subnet = []
        for i in range(self.num_cls_convs):
            cls_subnet.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        for i in range(self.num_reg_convs):
            bbox_subnet.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bbox_pred = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * 4,
            kernel_size=3,
            stride=1,
            padding=1)
        self.object_pred = nn.Conv2d(
            self.in_channels,
            self.num_base_priors,
            kernel_size=3,
            stride=1,
            padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

        # Use prior in model initialization to improve stability
        bias_cls = bias_init_with_prob(0.01)
        torch.nn.init.constant_(self.cls_score.bias, bias_cls)

    def forward_single(self, feature):
        cls_score = self.cls_score(self.cls_subnet(feature))
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)

        reg_feat = self.bbox_subnet(feature)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)

        # 这里是为了方便使用sigmoid而进行的操作,
        # 即sigmoid(normalized_cls_score) = sigmoid(cls)*sigmoid(obj)
        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(
            1. + torch.clamp(cls_score.exp(), max=INF) +
            torch.clamp(objectness.exp(), max=INF))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
        return normalized_cls_score, bbox_reg

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """计算Head的损失.

        Args:
            cls_scores (list[Tensor]): [[bs, na * nc, h, w],] * num_level
            bbox_preds (list[Tensor]): [[bs, na * 4, h, w],] * num_level
            gt_bboxes (list[Tensor]): [[num_gts, 4],] * bs.
            gt_labels (list[Tensor]): [[num_gts,],] * bs.
            img_metas (list[dict]): [{img_meta},] * bs.
            gt_bboxes_ignore (None | list[Tensor]): 计算损失时可以指定忽略哪些gt box.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == 1
        assert self.prior_generator.num_levels == 1

        device = cls_scores[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # [[[h * w * na, 4], ] * num_levels,] * bs
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        # 因为YOLOF是单个特征图,所以输出的anchor层级始终为第一个
        anchor_list = [anchors[0] for anchors in anchor_list]
        valid_flag_list = [valid_flags[0] for valid_flags in valid_flag_list]

        # [[bs, c, h, w],] * num_level -> [[num_level * h * w, c],] * bs
        cls_scores_list = levels_to_images(cls_scores)
        bbox_preds_list = levels_to_images(bbox_preds)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:  # batch幅图像中任意一个没有有效anchor
            return None
        (batch_labels, batch_label_weights, num_total_pos, num_total_neg,
         batch_bbox_weights, batch_pos_predicted_boxes,
         batch_target_boxes) = cls_reg_targets

        flatten_labels = batch_labels.reshape(-1)
        batch_label_weights = batch_label_weights.reshape(-1)
        cls_score = cls_scores[0].permute(0, 2, 3,
                                          1).reshape(-1, self.cls_out_channels)

        num_total_samples = (num_total_pos +
                             num_total_neg) if self.sampling else num_total_pos
        num_total_samples = reduce_mean(
            cls_score.new_tensor(num_total_samples)).clamp_(1.0).item()

        # cls loss
        loss_cls = self.loss_cls(
            cls_score,
            flatten_labels,
            batch_label_weights,  # 仅仅包括正负样本的权重,不计算忽略样本的权重
            avg_factor=num_total_samples)

        # reg loss
        if batch_pos_predicted_boxes.shape[0] == 0:
            # 没有正样本
            loss_bbox = batch_pos_predicted_boxes.sum() * 0
        else:
            loss_bbox = self.loss_bbox(
                batch_pos_predicted_boxes,
                batch_target_boxes,
                batch_bbox_weights.float(),  # 没有被pos_ignore_thr阈值忽略掉的正样本
                avg_factor=num_total_samples)

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """计算batch张图像中anchor的reg和cls的target.

        Args:
            cls_scores_list (list[Tensor])： [[h * w, na * nc], ] * bs
            bbox_preds_list (list[Tensor])： [[h * w, na * 4], ] * bs
            anchor_list (list[Tensor]): [[h * w * na, 4],] * bs.
            valid_flag_list (list[Tensor]): [[h * w * na, ], ] * bs.
            gt_bboxes_list (list[Tensor]): [[num_gts, 4], ] * bs.
            img_metas (list[dict]): [{img_meta},] * bs.
            gt_bboxes_ignore_list (list[Tensor]): 计算损失时可以指定忽略哪些gt box.
            gt_labels_list (list[Tensor]): [[num_gts,],] * bs.
            label_channels (int): 背景类.
            unmap_outputs (bool): 是否将有效anchor的reg和cls目标映射回原始anchor上.

        Returns:
            tuple: 通常返回一个包含学习目标的元组.

                - batch_labels (Tensor): 所有图像上的所有anchor的cls_target.
                    [bs, h * w * na]
                - batch_label_weights (Tensor): cls_target的权重(正负样本为1,其余为0)
                    [bs, h * w * na]
                - num_total_pos (int): 所有图像上的正样本数量.
                - num_total_neg (int): 所有图像上的负样本数量..
            additional_returns: 这块内容来自 `self._get_targets_single` 的用户定义返回.
                将在函数末尾处与上面的元组结合到一起返回
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # 计算每张图片的拟合目标
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, pos_inds_list, neg_inds_list,
         sampling_results_list) = results[:5]
        rest_results = list(results[5:])  # user-added return values
        # 如果batch张图片中任意一张图片上没有有效的anchor都直接返回None
        if any([labels is None for labels in all_labels]):
            return None
        # 整个batch图片上采集到的样本
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

        batch_labels = torch.stack(all_labels, 0)
        batch_label_weights = torch.stack(all_label_weights, 0)

        res = (batch_labels, batch_label_weights, num_total_pos, num_total_neg)
        for i, rests in enumerate(rest_results):  # user-added return values
            rest_results[i] = torch.cat(rests, 0)

        return res + tuple(rest_results)

    def _get_targets_single(self,
                            bbox_preds,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """计算单个图像中anchor的reg和cls的target.

        Args:
            bbox_preds (Tensor): head的reg输出, [h * w, na * 4]
            flat_anchors (Tensor): 整张图像的anchor,[h * w * na ,4]
            valid_flags (Tensor): flat_anchors对应的有效mask, [h * w * na,].
            gt_bboxes (Tensor): [num_gts, 4].
            gt_bboxes_ignore (Tensor): [num_ignored_gts, 4].
            img_meta (dict): 图像元信息.
            gt_labels (Tensor): [num_gts,].
            label_channels (int): 背景类.
            unmap_outputs (bool): 是否将有效anchor的reg和cls目标映射回原始anchor上.

        Returns:
            tuple:
                labels (Tensor): anchor的cls_target, (h * w * na, ).
                label_weights (Tensor): anchor的cls_weight, (h * w * na, ).
                pos_inds (Tensor): 正样本索引.
                neg_inds (Tensor): 负样本索引.
                sampling_result (obj:`SamplingResult`): 采样结果.
                pos_bbox_weights (Tensor): 用于计算box分支loss的权重,
                    [self.match_times * 2 * num_gt,].
                pos_predicted_boxes (Tensor): 用于计算box分支loss的boxes预测值,
                    [self.match_times * 2 * num_gt, 4].
                pos_target_boxes (Tensor): 用于计算box分支loss的boxes目标值,
                    [self.match_times * 2 * num_gt, 4].
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():  # 如果该张图像上没有一个有效anchor
            return (None, ) * 8
        # 对anchors分配gt然后进行采样
        anchors = flat_anchors[inside_flags, :]
        bbox_preds = bbox_preds.reshape(-1, 4)
        bbox_preds = bbox_preds[inside_flags, :]

        # decoded bbox
        decoder_bbox_preds = self.bbox_coder.decode(anchors, bbox_preds)
        assign_result = self.assigner.assign(
            decoder_bbox_preds, anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)

        # 因为正样本(anchor+box)iou中可能有小于pos_ignore_thr的忽略样本存在,
        # 该值就是构造一个正样本中"有效正样本"的mask.其中伪正样本为False,其余为True.
        # shape为[match_times * 2 * num_gt,]
        pos_bbox_weights = assign_result.get_extra_property('pos_idx')
        # 指bbox_pred[indexes],其中indexes为anchor与pred_box两者与gt box的
        # L1距离最近的match_time * 2个索引cat到一起
        # [self.match_times * 2 * num_gt, 4],此时还没过滤掉忽略样本
        pos_predicted_boxes = assign_result.get_extra_property(
            'pos_predicted_boxes')
        # shape同上,值为gt_box[torch.arange(num_gt).repeat(2*match_time)]
        pos_target_boxes = assign_result.get_extra_property('target_boxes')

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if gt_labels is None:
                # gt_labels只有在rpn阶段才为None,此时前景为0,背景为1
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
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # cls_target 默认为背景类
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)

        return (labels, label_weights, pos_inds, neg_inds, sampling_result,
                pos_bbox_weights, pos_predicted_boxes, pos_target_boxes)
