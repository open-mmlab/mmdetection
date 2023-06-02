# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.models.layers import multiclass_nms
from mmdet.models.losses import accuracy
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances, multi_apply
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from mmdet.utils import ConfigType, InstanceList, OptMultiConfig


@MODELS.register_module()
class BBoxHead(BaseModule):
    """最简单的 RoI Head,只有两个 fc 层分别用于分类和回归."""

    def __init__(self,
                 with_avg_pool: bool = False,
                 with_cls: bool = True,
                 with_reg: bool = True,
                 roi_feat_size: int = 7,
                 in_channels: int = 256,
                 num_classes: int = 80,
                 bbox_coder: ConfigType = dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 predict_box_type: str = 'hbox',
                 reg_class_agnostic: bool = False,
                 reg_decoded_bbox: bool = False,
                 reg_predictor_cfg: ConfigType = dict(type='Linear'),
                 cls_predictor_cfg: ConfigType = dict(type='Linear'),
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.predict_box_type = predict_box_type
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area  # 256 * 7 * 7
        if self.with_cls:
            # need to add background class
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = num_classes + 1
            cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
            cls_predictor_cfg_.update(
                in_features=in_channels, out_features=cls_channels)
            self.fc_cls = MODELS.build(cls_predictor_cfg_)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if reg_class_agnostic else \
                box_dim * num_classes
            reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_predictor_cfg_, (dict, ConfigDict)):
                reg_predictor_cfg_.update(
                    in_features=in_channels, out_features=out_dim_reg)
            self.fc_reg = MODELS.build(reg_predictor_cfg_)
        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            if self.with_cls:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.01, override=dict(name='fc_cls'))
                ]
            if self.with_reg:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.001, override=dict(name='fc_reg'))
                ]

    # TODO: Create a SeasawBBoxHead to simplified logic in BBoxHead
    @property
    def custom_cls_channels(self) -> bool:
        """get custom_cls_channels from loss_cls."""
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    # TODO: Create a SeasawBBoxHead to simplified logic in BBoxHead
    @property
    def custom_activation(self) -> bool:
        """get custom_activation from loss_cls."""
        return getattr(self.loss_cls, 'custom_activation', False)

    # TODO: Create a SeasawBBoxHead to simplified logic in BBoxHead
    @property
    def custom_accuracy(self) -> bool:
        """get custom_accuracy from loss_cls."""
        return getattr(self.loss_cls, 'custom_accuracy', False)

    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all
                  scale levels, each is a 4D-tensor, the channels number
                  is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all
                  scale levels, each is a 4D-tensor, the channels number
                  is num_base_priors * 4.
        """
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def _get_targets_single(self, pos_priors: Tensor, neg_priors: Tensor,
                            pos_gt_bboxes: Tensor, pos_gt_labels: Tensor,
                            cfg: ConfigDict) -> tuple:
        """根据采样结果计算单张图片中正负样本对应的cls target与reg target.

        Args:
            pos_priors (Tensor): [num_pos, 4], 正样本
            neg_priors (Tensor): [num_neg, 4], 负样本
            pos_gt_bboxes (Tensor): [num_pos, 4], 正样本对应的的gt box
            pos_gt_labels (Tensor): [num_pos, ], 正样本对应的的gt label
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: 单张图片中正负样本的拟合目标:

                - labels(Tensor): target cls, [num_pos+num_neg, ].
                - label_weights(Tensor): target cls的权重, [num_pos+num_neg, ].
                - bbox_targets(Tensor): target box, [num_pos+num_neg, 4]. xyxy格式.
                - bbox_weights(Tensor):target box的权重, [num_pos+num_neg, 4].
        """
        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_priors.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        reg_dim = pos_gt_bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        label_weights = pos_priors.new_zeros(num_samples)
        bbox_targets = pos_priors.new_zeros(num_samples, reg_dim)
        bbox_weights = pos_priors.new_zeros(num_samples, reg_dim)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_priors, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = get_box_tensor(pos_gt_bboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict,
                    concat: bool = True) -> tuple:
        """根据sampling_results计算一个batch中所有样本对应的gt box.

        与 bbox_head 中的实现几乎相同, 我们将附加参数 pos_inds_list 和
        neg_inds_list 传递给 `_get_target_single` 函数.

        Args:
            sampling_results (List[obj:SamplingResult]): batch幅图像的样本采样结果.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): 是否将batch张图像的结果cat到一起.

        Returns:
            Tuple[Tensor]: roi对应的cls target与reg target.

            - labels (list[Tensor],Tensor): [num_roi, ] * bs.
                  如果 concat为True时,则为[bs * num_roi, ]. 下同
            - label_weights (list[Tensor]): [num_roi, ] * bs.
            - bbox_targets (list[Tensor],Tensor): [num_roi, 4] * bs, xyxy格式.
            - bbox_weights (list[tensor],Tensor): [num_roi, 4] * bs.
        """
        pos_priors_list = [res.pos_priors for res in sampling_results]
        neg_priors_list = [res.neg_priors for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_targets_single,
            pos_priors_list,
            neg_priors_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        rois: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigDict,
                        concat: bool = True,
                        reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): 根据正负样本输出的cls score, [bs*(pos_num+neg_num), nc]
            bbox_pred (Tensor): 同上,reg score, [bs*(pos_num+neg_num), nc*4], xyxy格式
            rois (Tensor): rpn或r-cnn提供的roi,然后根据与gt box匹配与采样后的结果提取出
                [bs*(pos_num+neg_num), 5] 5 -> [img_index, x1, y1, x2, y2]
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): 是否覆盖Loss类初始化中的reduction.
                可选"none","mean", "sum".

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """

        cls_reg_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)
        losses = self.loss(
            cls_score,
            bbox_pred,
            rois,
            *cls_reg_targets,
            reduction_override=reduction_override)

        # cls_reg_targets is only for cascade rcnn
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): 根据正负样本输出的cls score, [bs*(pos_num+neg_num), nc]
            bbox_pred (Tensor): 同上,reg score, [bs*(pos_num+neg_num), nc*4], xyxy格式
            rois (Tensor): rpn或r-cnn提供的roi,然后根据与gt box匹配与采样后的结果提取出
                [bs*(pos_num+neg_num), 5] 5 -> [img_index, x1, y1, x2, y2]
            labels (Tensor): rois对应的gt label[bs*(pos_num+neg_num), ]
            label_weights (Tensor): rois对应的cls weight[bs*(pos_num+neg_num), ]
            bbox_targets (Tensor): rois对应的gt box[bs*(pos_num+neg_num), 4],xyxy格式.
            bbox_weights (Tensor): rois对应的reg weight[bs*(pos_num+neg_num), 4]
            reduction_override (str, optional): 是否覆盖Loss类初始化中的reduction.
                可选"none","mean", "sum".

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,  # 该参数仅在最终reduction=mean时才生效
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 前景:[0, nc), 背景:nc
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # prior在不同的地方寓意不同,在一阶段(或者二阶段的rpn中)
                    # 指代anchor(如果是anchor base的网络的话),其余地方代指 roi
                    # 该参数为True时,意为需要根据prior与reg结合.
                    # 计算得到box的绝对坐标与gt box一同送入损失函数进行计算loss
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        return losses

    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        bbox_preds: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: Optional[ConfigDict] = None,
                        rescale: bool = False) -> InstanceList:
        """将网络的单张图片输出转换为最终检测box与label.

        Args:
            rois (tuple[Tensor]): 将要进行修正的box. [[num_boxes, 5], ] * nl.
               5 -> (batch_index, x1, y1, x2, y2).
            cls_scores (tuple[Tensor]): roi cls, [[num_boxes, nc + 1], ] * nl.
            bbox_preds (tuple[Tensor]): roi reg, [[num_boxes, nc * 4], ] * nl.
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (obj:`ConfigDict`, optional): `test_cfg` of R-CNN.
                Defaults to None.
            rescale (bool): 是否将box进行缩放回原始原始图像尺寸.

        Returns:
            list[:obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): [num_instances, ]
                - labels (Tensor): [num_instances, ].
                - bboxes (Tensor): [num_instances, 4], xyxy格式.
        """
        assert len(cls_scores) == len(bbox_preds)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                bbox_pred=bbox_preds[img_id],
                img_meta=img_meta,
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg)
            result_list.append(results)

        return result_list

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """将网络的单张图片输出转换为最终检测box与label.

        Args:
            roi (Tensor): 将要进行修正的box. [num_boxes, 5].
               5 -> (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): roi cls, [num_boxes, nc + 1].
            bbox_pred (Tensor): roi reg, [num_boxes, nc * 4].
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): 是否将box进行缩放回原始原始图像尺寸.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): [num_instances, ]
                - labels (Tensor): [num_instances, ].
                - bboxes (Tensor): [num_instances, 4], xyxy格式.
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]

        # 一些loss,比如(Seesaw loss..) 可能有自定义激活函数
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results

    def refine_bboxes(self, sampling_results: Union[List[SamplingResult],
                                                    InstanceList],
                      bbox_results: dict,
                      batch_img_metas: List[dict]) -> InstanceList:
        """在训练期间微调box. 令N=bs*(pos_num+neg_num)

        Args:
            sampling_results (List[:obj:`SamplingResult`] or
                List[:obj:`InstanceData`]): Sampling results.
                :obj:`SamplingResult` is the real sampling results
                calculate from bbox_head, while :obj:`InstanceData` is
                fake sampling results, e.g., in Sparse R-CNN or QueryInst, etc.
            bbox_results (dict): Usually is a dictionary with keys:

                - `cls_score` (Tensor): roi cls, [N, nc].
                - `bbox_pred` (Tensor): roi reg, [N, 4] or [N, nc*4].
                - `rois` (Tensor): [N, 5], [img_ind, x1, y1, x2, y2]
                - `bbox_targets` (tuple):  Ground truth for proposals in a
                  single image. Containing the following list of Tensors:
                  (labels, label_weights, bbox_targets, bbox_weights)
            batch_img_metas (List[dict]): List of image information.

        Returns:
            list[:obj:`InstanceData`]: Refined bboxes of each image.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import numpy as np
            >>> from mmdet.models.task_modules.samplers.
            ... sampling_result import random_boxes
            >>> from mmdet.models.task_modules.samplers import SamplingResult
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            ... batch_img_metas = [{'img_shape': (scale, scale)}
            >>>                     for _ in range(n_img)]
            >>> sampling_results = [SamplingResult.random(rng=10)
            ...                     for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 81, (scale,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> cls_score = torch.randn((scale, 81))
            ... # For each image, pretend random positive boxes are gts
            >>> bbox_targets = (labels, None, None, None)
            ... bbox_results = dict(rois=rois, bbox_pred=bbox_preds,
            ...                     cls_score=cls_score,
            ...                     bbox_targets=bbox_targets)
            >>> bboxes_list = self.refine_bboxes(sampling_results,
            ...                                  bbox_results,
            ...                                  batch_img_metas)
            >>> print(bboxes_list)
        """
        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        # bbox_targets is a tuple
        labels = bbox_results['bbox_targets'][0]
        cls_scores = bbox_results['cls_score']
        rois = bbox_results['rois']
        bbox_preds = bbox_results['bbox_pred']
        if self.custom_activation:
            # TODO: Create a SeasawBBoxHead to simplified logic in BBoxHead
            cls_scores = self.loss_cls.get_activation(cls_scores)
        if cls_scores.numel() == 0:
            return None
        if cls_scores.shape[-1] == self.num_classes + 1:
            # remove background class
            cls_scores = cls_scores[:, :-1]
        elif cls_scores.shape[-1] != self.num_classes:
            raise ValueError('The last dim of `cls_scores` should equal to '
                             '`num_classes` or `num_classes + 1`,'
                             f'but got {cls_scores.shape[-1]}.')
        # 当预测label为背景类时,将其改为其他最大类别概率对应的类别.
        labels = torch.where(labels == self.num_classes, cls_scores.argmax(1),
                             labels)

        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(batch_img_metas)

        results_list = []
        for i in range(len(batch_img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()  # 当前图像的正负样本数量

            # 归属于某张图片的 roi [num_pos+num_neg, 4]
            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = batch_img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # 过滤掉box中的gt
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep
            results = InstanceData(bboxes=bboxes[keep_inds.type(torch.bool)])
            results_list.append(results)

        return results_list

    def regress_by_class(self, priors: Tensor, label: Tensor,
                         bbox_pred: Tensor, img_meta: dict) -> Tensor:
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            priors (Tensor): 来自`rpn_head`或roi head最后一阶段的bbox_head的prior,
                [sample.num, 4].
            label (Tensor): 仅在reg_class_agnostic为False时使用, [sample.num, ].
            bbox_pred (Tensor): box head的reg输出,对roi再次修正 reg_class_agnostic
                为True时,表示当前box仅仅是前景[sample.num, 4],仅有一个(前景).
                为False时,表示当前box在不同类别上独立回归[sample.num, nc * 4],所以就根据
                最大的cls score所对应的cls ind(上面的label参数)来获取相应的reg和prior结合
                计算得到box.
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        reg_dim = self.bbox_coder.encode_size
        if not self.reg_class_agnostic:
            label = label * reg_dim
            inds = torch.stack([label + i for i in range(reg_dim)], 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size()[1] == reg_dim

        max_shape = img_meta['img_shape']
        regressed_bboxes = self.bbox_coder.decode(
            priors, bbox_pred, max_shape=max_shape)
        return regressed_bboxes
