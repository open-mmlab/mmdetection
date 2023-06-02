# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import BaseBoxes, cat_boxes, get_box_tensor
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig)
from ..task_modules.prior_generators import (AnchorGenerator,
                                             anchor_inside_flags)
from ..task_modules.samplers import PseudoSampler
from ..utils import images_to_levels, multi_apply, unmap
from .base_dense_head import BaseDenseHead


@MODELS.register_module()
class AnchorHead(BaseDenseHead):
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
        init_cfg (dict or list[dict], optional): 参数初始化配置字典.
    """  # noqa: W605

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        anchor_generator: ConfigType = dict(
            type='AnchorGenerator',
            scales=[8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder: ConfigType = dict(
            type='DeltaXYWHBBoxCoder',
            clip_border=True,
            target_means=(.0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0)),
        reg_decoded_bbox: bool = False,
        loss_cls: ConfigType = dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox: ConfigType = dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = dict(
            type='Normal', layer='Conv2d', std=0.01)
    ) -> None:
        super().__init__(init_cfg=init_cfg)
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

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            if train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'], default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler(context=self)

        self.fp16_enabled = False

        self.prior_generator = TASK_UTILS.build(anchor_generator)

        # 通常每个层级的基础anchor数量是相同的,但是SSDHead的各层级基础anchor则是一个列表,
        # 因为它不同层级特征点上的anchor数量不同,所以无法用整数表示.
        # self.prior_generator.num_base_priors是指各个层级上先验数量,是个列表
        # 而本类中的self.num_base_priors则指单一层级上的先验数量,是个整数
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_cls = nn.Conv2d(self.in_channels,
                                  self.num_base_priors * self.cls_out_channels,
                                  1)
        reg_dim = self.bbox_coder.encode_size
        self.conv_reg = nn.Conv2d(self.in_channels,
                                  self.num_base_priors * reg_dim, 1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """单一层级特征图的前向传播.

        Args:
            x (Tensor): 单一层级特征图.

        Returns:
            tuple:
                cls_score (Tensor): 单层级的cls_scores, [bs, na*nc, h, w]
                bbox_pred (Tensor): 单层级的cls_scores, [bs, na*4, h, w]
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """对来自上游网络的特征图进行前向传播.

        Args:
            x (tuple[Tensor]): 来自上游网络的特征图,每个都是 4维 张量.

        Returns:
            tuple: tuple(cls_scores, reg_predict).

                - cls_scores (list[Tensor]): 所有层级的cls_scores,
                    [[bs, na*nc, h, w], ] * num_level
                - reg_predict (list[Tensor]): 所有层级的box_reg,
                    [[bs, na*4, h, w], ] * num_level
        """
        return multi_apply(self.forward_single, x)

    def get_anchors(self,
                    featmap_sizes: List[tuple],
                    batch_img_metas: List[dict],
                    device: Union[torch.device, str] = 'cuda') \
            -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
        """根据batch幅图像(直接复制)、多层级特征图尺寸获取anchors.

        对于每副图像,collate阶段会将较小图片进行padding,这些padding区域就称为无效区域.
        anchor生成时是基于输入图像尺寸进行生成的,当然也会在无效区域生成一些anchor.
        因此过滤这些无效区域的anchor以使其避免参与后续的相关计算是有必要的.
        值得注意的是这与anchor本身的尺寸是无关的,极端点来看只要在有效区域,
        哪怕这个anchor尺寸非常离谱甚至超出了图像范围也是"有效anchor".
        只要在无效区域,哪怕这个anchor与gt box的iou很高也是"无效anchor"
        以及anchor生成时默认以特征点左上角为中心点的.
        Args:
            featmap_sizes (list[tuple]): 多层级特征图尺寸.
            img_metas (list[dict]): 图像元信息.
            device (torch.device | str): 生成anchor的设备

        Returns:
            tuple:
                anchor_list (list[list[Tensor]]): 多张图像的anchor.
                    [[[h * w * na, 4], ] * num_levels,] * bs
                valid_flag_list (list[list[Tensor]]): 各个特征图上有效anchor的mask.
                    [[[h * w * na, ], ] * num_levels, ] * bs
        """
        num_imgs = len(batch_img_metas)

        # 由于一个batch中所有图像的特征图大小相同,我们这里只计算一次anchor,然后复制即可
        # [[h * w * na, 4], ] * num_levels
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        valid_flag_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def _get_targets_single(self,
                            flat_anchors: Union[Tensor, BaseBoxes],
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """计算单张图像上anchor的回归和分类拟合目标.

        Args:
            flat_anchors (Tensor or :obj:`BaseBoxes`): [nl * h * w * na, 4]
            valid_flags (Tensor): 同上,有效anchor的mask, [nl * h * w * na,]
            gt_instances (:obj:`InstanceData`): 真实的标注信息. 包含bboxes,labels属性.
            img_meta (dict): 单张图片的元信息
            gt_instances_ignore (:obj:`InstanceData`, optional): 忽略的标注信息. 包含bboxes属性.
            unmap_outputs (bool): 是否将计算出的有效anchor的label、target、weight映射回原始anchor上

        Returns:
            tuple:
                - labels (Tensor): 各层级上的cls_target.
                    unmap_outputs为True时,[num_levels * h * w * na, ]
                    unmap_outputs为False时,[num_valid_anchors, ]
                - label_weights (Tensor): 各层级上的cls_target权重.
                - bbox_targets (Tensor): 各层级上的reg_target.
                - bbox_weights (Tensor): 各层级上的target_reg权重,
                - pos_inds (Tensor): 该张图片中的正样本索引.
                - neg_inds (Tensor): 该张图片中的负样本索引.
                - sampling_result (:obj:`SamplingResult`): 采样结果.
        """
        # flat_anchors -> [num_levels*(h * w * na), 4]
        # valid_flags过滤了在collact阶段填充的多余像素区域生成的anchor,而这一步是过滤掉超出图像边界的一些anchor
        # 此处的图像边界指的是在进行collact操作之前时的图像尺寸.
        # self.train_cfg.allowed_border默认为-1,即不进行过滤.
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags]

        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)
        # No sampling is required except for RPN and
        # Guided Anchoring algorithms
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        target_dim = gt_instances.bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        bbox_targets = anchors.new_zeros(num_valid_anchors, target_dim)
        bbox_weights = anchors.new_zeros(num_valid_anchors, target_dim)

        # TODO: Considering saving memory, is it necessary to be long?
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # `bbox_coder.encode` accepts tensor or box type inputs and generates
        # tensor targets. If regressing decoded boxes, the code will convert
        # box type `pos_bbox_targets` to tensor.
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
                pos_bbox_targets = get_box_tensor(pos_bbox_targets)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
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
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True,
                    return_sampling_results: bool = False) -> tuple:
        """计算多张图像中anchor的回归和分类目标.

        Args:
            anchor_list (list[list[Tensor]]): [[[h * w * na, 4], ] * nl] * bs.
            valid_flag_list (list[list[Tensor]]): [[[h * w * na, ], ] * nl] * bs
            batch_gt_instances (list[:obj:`InstanceData`]): 真实的标注信息.
                它通常包括``bboxes`` 和``labels`` 属性.
            batch_img_metas (list[dict]): batch张图片的元信息.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                忽略的标注信息. 它通常包含``bboxes``属性.
            unmap_outputs (bool): 是否将输出映射回原始anchor上.
            return_sampling_results (bool): 是否返回采样结果.

        Returns:
            tuple: 通常返回一个包含网络拟合目标的元组.

                - labels_list (list[Tensor]): 每个层级上对应的label.
                    [[bs, h * w * na, ], ] * nl
                - label_weights_list (list[Tensor]): 每个层级上的label权重.
                    [[bs, h * w * na, ], ] * nl
                - bbox_targets_list (list[Tensor]): 每个层级上特征图需要回归的目标.
                    [[bs, h * w * na, 4], ] * nl
                - bbox_weights_list (list[Tensor]): 每个层级上特征图回归的权重.
                    [[bs, h * w * na, 4], ] * nl
                - avg_factor (int): 用于平均loss的平均因子. 当使用采样方法时,
                    avg_factor 通常是正负先验之和. 当使用PseudoSampler方法时,
                    avg_factor 通常是正先验之和.

            额外返回值: 此值来自 `self._get_targets_single` 的用户定义返回.
                这些返回值目前被细化为每个特征图的属性(即具有 HxW 维度).它将与原始返回值
                合并到一起返回.
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        # 单张图片的各个层级上的anchor总数量,(向上取整),以RetinaNet-FPN为例
        # [h/8*w/8*9, h/16*w/16*9, h/32*w/32*9, h/64*w/64*9, h/128*w/128*9]
        # [43200, 10800, 2700, 720, 180],其中网络输入batch尺寸(h,w)=(480, 640)
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # 将一张图片上的所有anchor全部cat到一起
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # 计算batch张图像的拟合目标
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # Get `avg_factor` of all images, which calculate in `SamplingResult`.
        # When using sampling method, avg_factor is usually the sum of
        # positive and negative priors. When using `PseudoSampler`,
        # `avg_factor` is usually equal to the number of positive priors.
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        # update `_raw_positive_infos`, which will be used when calling
        # `get_positive_infos`.
        self._raw_positive_infos.update(sampling_results=sampling_results_list)
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, avg_factor)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            anchors: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor, avg_factor: int) -> tuple:
        """计算单个层级的损失.

        Args:
            cls_score (Tensor): 每层级输出的类别置信度 [bs, na * nc, h, w].
            bbox_pred (Tensor): 每层级输出的box/loc  [bs, na * 4, h, w].
            anchors (Tensor): 每层级的基础anchor     [bs, h * w * na, 4].
            labels (Tensor): 每层级的类别拟合目标     [bs, h * w * na,].
            label_weights (Tensor): 每层级的类别权重 [bs, h * w * na,].
            bbox_targets (Tensor): 每层级的回归拟合目标  [bs, h * w * na, 4].
            bbox_weights (Tensor): 每层级的回归权重  [bs, h * w * na, 4].
            avg_factor (int): 用于平均loss的平均因子.

        Returns:
            dict[str, Tensor]: 分类和回归损失.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)
        # regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1,
                                                 self.bbox_coder.encode_size)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        return loss_cls, loss_bbox

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
            cls_scores (list[Tensor]): 每个层级的box score
                 [[bs, na * nc, h, w],] * nl
            bbox_preds (list[Tensor]): 每个层级的box reg
                [[bs, na * 4, h, w],] * nl
            batch_gt_instances (list[:obj:`InstanceData`]): batch幅图像的实例信息
                通常包含``bboxes`` and ``labels``属性.
            batch_img_metas (list[dict]): batch张图片的元信息.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                batch幅图像中忽略的实例信息通常包含``bboxes``属性.

        Returns:
            dict: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        # 获取batch图像上的所有anchor,以及有效anchor的mask(在collate阶段填充的区域被视为无效区域)
        # [[[h * w * na, 4], ] * num_levels, ] * bs, [[[h * w * na, ], ] * num_levels] * bs.
        # 注意以上两个变量,前者只要计算一张图像上的anchor然后复制B次,因为同一batch下所有图像尺寸一致
        # 而后者依据每幅图片的尺寸不同,anchor有效区域也可能不同,需要具体计算出每张图像的有效mask.
        # 以及feat_h*feat_w是随着层级(level)变化而变化的
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        # 利用提供的anchor将其与每张图片上的target计算IOU以及利用相应的thr来分配正负样本
        # 接着采取sample采样正负样本,主要用来筛选合适的正负样本用来训练.同时为他们分配最佳gt和label
        # 最后将数据进行调整以和网络前向传播结果保持一致,方便loss计算
        cls_reg_targets = self.get_targets(  # [level0,level1,...]
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor) = cls_reg_targets

        # 各层级anchor数量列表
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # 现将每个图像上的anchor合并到一起,然后由img级别映射到level级别,该操作在get_targets
        # 方法最后部分也存在.之所以这么做的原因是为了和网络输出shape保持一致,方便计算loss
        concat_anchor_list = []
        for i in range(len(anchor_list)):  # 现将每个图像上的anchor合并到一起
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
