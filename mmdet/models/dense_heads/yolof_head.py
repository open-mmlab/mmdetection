# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, is_norm
from mmengine.model import bias_init_with_prob, constant_init, normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList, OptInstanceList, reduce_mean
from ..task_modules.prior_generators import anchor_inside_flags
from ..utils import levels_to_images, multi_apply, unmap
from .anchor_head import AnchorHead

INF = 1e8


@MODELS.register_module()
class YOLOFHead(AnchorHead):
    """Detection Head of `YOLOF <https://arxiv.org/abs/2103.09460>`_

    Args:
        num_classes (int): 检测类别数量
        in_channels (list[int]): 每个scale的输入通道数量.
        cls_num_convs (int): cls分支的卷积数量.默认为2.
        reg_num_convs (int): reg分支的卷积数量.默认为4.
        norm_cfg (:obj:`ConfigDict` or dict): 构造和配置norm层的字典.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: List[int],
                 num_cls_convs: int = 2,
                 num_reg_convs: int = 4,
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 **kwargs) -> None:
        self.num_cls_convs = num_cls_convs
        self.num_reg_convs = num_reg_convs
        self.norm_cfg = norm_cfg
        super().__init__(
            num_classes=num_classes, in_channels=in_channels, **kwargs)

    def _init_layers(self) -> None:
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

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

        # Use prior in model initialization to improve stability
        bias_cls = bias_init_with_prob(0.01)
        torch.nn.init.constant_(self.cls_score.bias, bias_cls)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                normalized_cls_score (Tensor): Normalized Cls scores for a \
                    single scale level, the channels number is \
                    num_base_priors * num_classes.
                bbox_reg (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """
        cls_score = self.cls_score(self.cls_subnet(x))
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)

        reg_feat = self.bbox_subnet(x)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)

        # 这里是为了方便后续使用sigmoid而进行的操作,
        # 即sigmoid(normalized_cls_score) = sigmoid(cls)*sigmoid(obj)
        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(
            1. + torch.clamp(cls_score.exp(), max=INF) +
            torch.clamp(objectness.exp(), max=INF))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
        return normalized_cls_score, bbox_reg

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): [[bs, na * nc, h, w],] * nl
            bbox_preds (list[Tensor]): [[bs, na * 4, h, w],] * nl
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
        assert len(cls_scores) == 1
        assert self.prior_generator.num_levels == 1

        device = cls_scores[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # [[[h * w * na, 4], ] * nl,] * bs
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        # 因为YOLOF是单个特征图,所以输出的anchor层级始终为第一个
        anchor_list = [anchors[0] for anchors in anchor_list]
        valid_flag_list = [valid_flags[0] for valid_flags in valid_flag_list]

        cls_scores_list = levels_to_images(cls_scores)
        bbox_preds_list = levels_to_images(bbox_preds)

        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        if cls_reg_targets is None:  # bs幅图像中存在没有有效anchor的图像
            return None
        (batch_labels, batch_label_weights, avg_factor, batch_bbox_weights,
         batch_pos_predicted_boxes, batch_target_boxes) = cls_reg_targets

        flatten_labels = batch_labels.reshape(-1)
        batch_label_weights = batch_label_weights.reshape(-1)
        cls_score = cls_scores[0].permute(0, 2, 3,
                                          1).reshape(-1, self.cls_out_channels)

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        # cls loss
        loss_cls = self.loss_cls(
            cls_score,
            flatten_labels,
            batch_label_weights,  # 仅仅包括正负样本的权重,不计算忽略样本的权重
            avg_factor=avg_factor)

        # reg loss
        if batch_pos_predicted_boxes.shape[0] == 0:
            # 没有正样本
            loss_bbox = batch_pos_predicted_boxes.sum() * 0
        else:
            loss_bbox = self.loss_bbox(
                batch_pos_predicted_boxes,
                batch_target_boxes,
                batch_bbox_weights.float(),  # 没有被pos_ignore_thr阈值忽略掉的正样本
                avg_factor=avg_factor)

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def get_targets(self,
                    cls_scores_list: List[Tensor],
                    bbox_preds_list: List[Tensor],
                    anchor_list: List[Tensor],
                    valid_flag_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True):
        """计算batch张图像中anchor的reg和cls的target.

        Args:
            cls_scores_list (list[Tensor])： [[h * w, na * nc], ] * bs
            bbox_preds_list (list[Tensor])： [[h * w, na * 4], ] * bs
            anchor_list (list[Tensor]): [[h * w * na, 4],] * bs.
            valid_flag_list (list[Tensor]): [[h * w * na, ], ] * bs.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): 是否将有效anchor的reg和cls目标映射回原始anchor上.

        Returns:
            tuple: 通常返回一个包含学习目标的元组.

                - batch_labels (Tensor): 所有图像上的所有anchor的cls_target.
                    [bs, h * w * na]
                - batch_label_weights (Tensor): cls_target的权重(正负样本为1,其余为0)
                    [bs, h * w * na]
                - num_total_pos (int): 所有图像上的正样本数量.
                - num_total_neg (int): 所有图像上的负样本数量.
            additional_returns: 这块内容来自 `self._get_targets_single` 的用户定义返回.
                将在函数末尾处与上面的元组结合到一起返回
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # 计算每张图片的拟合目标
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        results = multi_apply(
            self._get_targets_single,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, pos_inds, neg_inds,
         sampling_results_list) = results[:5]
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        rest_results = list(results[5:])  # user-added return values

        batch_labels = torch.stack(all_labels, 0)
        batch_label_weights = torch.stack(all_label_weights, 0)

        res = (batch_labels, batch_label_weights, avg_factor)
        for i, rests in enumerate(rest_results):  # user-added return values
            rest_results[i] = torch.cat(rests, 0)

        return res + tuple(rest_results)

    def _get_targets_single(self,
                            bbox_preds: Tensor,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """计算单个图像中anchor的reg和cls的target.

        Args:
            bbox_preds (Tensor): head的reg输出, [h * w, na * 4]
            flat_anchors (Tensor): 整张图像的anchor,[h * w * na ,4]
            valid_flags (Tensor): flat_anchors对应的有效mask, [h * w * na,].
            gt_bboxes (Tensor): [num_gts, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
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
                pos_target_boxes (Tensor): 于计算box分支loss的boxes目标值,
                    [self.match_times * 2 * num_gt, 4].
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                '如果该张图像上没有一个有效anchor. 请检查图像尺寸和anchor尺寸,'
                '或设置``allowed_border``为 -1以忽略anchor边界限制.')

        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        bbox_preds = bbox_preds.reshape(-1, 4)
        bbox_preds = bbox_preds[inside_flags, :]

        # decoded bbox
        decoder_bbox_preds = self.bbox_coder.decode(anchors, bbox_preds)
        pred_instances = InstanceData(
            priors=anchors, decoder_priors=decoder_bbox_preds)
        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)

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

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)
        num_valid_anchors = anchors.shape[0]
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
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
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # cls_target 默认为背景类
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)

        return (labels, label_weights, pos_inds, neg_inds, sampling_result,
                pos_bbox_weights, pos_predicted_boxes, pos_target_boxes)
