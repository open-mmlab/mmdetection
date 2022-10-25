# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, ModuleList
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, InstanceList, MultiConfig,
                         OptInstanceList, OptMultiConfig)
from ..task_modules.assigners import RegionAssigner
from ..task_modules.samplers import PseudoSampler
from ..utils import (images_to_levels, multi_apply, select_single_mlvl,
                     unpack_gt_instances)
from .base_dense_head import BaseDenseHead
from .rpn_head import RPNHead


class AdaptiveConv(BaseModule):
    """AdaptiveConv used to adapt the sampling location with the anchors.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple[int]): Size of the conv kernel.
            Defaults to 3.
        stride (int or tuple[int]): Stride of the convolution. Defaults to 1.
        padding (int or tuple[int]): Zero-padding added to both sides of
            the input. Defaults to 1.
        dilation (int or tuple[int]): Spacing between kernel elements.
            Defaults to 3.
        groups (int): Number of blocked connections from input channels to
            output channels. Defaults to 1.
        bias (bool): If set True, adds a learnable bias to the output.
            Defaults to False.
        adapt_type (str): Type of adaptive conv, can be either ``offset``
            (arbitrary anchors) or 'dilation' (uniform anchor).
            Defaults to 'dilation'.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or \
            list[dict]): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 1,
        dilation: Union[int, Tuple[int]] = 3,
        groups: int = 1,
        bias: bool = False,
        adapt_type: str = 'dilation',
        init_cfg: MultiConfig = dict(
            type='Normal', std=0.01, override=dict(name='conv'))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert adapt_type in ['offset', 'dilation']
        self.adapt_type = adapt_type

        assert kernel_size == 3, 'Adaptive conv only supports kernels 3'
        if self.adapt_type == 'offset':
            assert stride == 1 and padding == 1 and groups == 1, \
                'Adaptive conv offset mode only supports padding: {1}, ' \
                f'stride: {1}, groups: {1}'
            self.conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                groups=groups,
                bias=bias)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=dilation,
                dilation=dilation)

    def forward(self, x: Tensor, offset: Tensor) -> Tensor:
        """Forward function."""
        if self.adapt_type == 'offset':
            N, _, H, W = x.shape
            assert offset is not None
            assert H * W == offset.shape[1]
            # reshape [N, NA, 18] to (N, 18, H, W)
            offset = offset.permute(0, 2, 1).reshape(N, -1, H, W)
            offset = offset.contiguous()
            x = self.conv(x, offset)
        else:
            assert offset is None
            x = self.conv(x)
        return x


@MODELS.register_module()
class StageCascadeRPNHead(RPNHead):
    """Stage of CascadeRPNHead.

    Args:
        in_channels (int): Number of channels in the input feature map.
        anchor_generator (:obj:`ConfigDict` or dict): anchor generator config.
        adapt_cfg (:obj:`ConfigDict` or dict): adaptation config.
        bridged_feature (bool): whether update rpn feature. Defaults to False.
        with_cls (bool): whether use classification branch. Defaults to True.
        init_cfg :obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 anchor_generator: ConfigType = dict(
                     type='AnchorGenerator',
                     scales=[8],
                     ratios=[1.0],
                     strides=[4, 8, 16, 32, 64]),
                 adapt_cfg: ConfigType = dict(type='dilation', dilation=3),
                 bridged_feature: bool = False,
                 with_cls: bool = True,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        self.with_cls = with_cls
        self.anchor_strides = anchor_generator['strides']
        self.anchor_scales = anchor_generator['scales']
        self.bridged_feature = bridged_feature
        self.adapt_cfg = adapt_cfg
        super().__init__(
            in_channels=in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

        # override sampling and sampler
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            # use PseudoSampler when sampling is False
            if self.train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'], default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler(context=self)

        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal', std=0.01, override=[dict(name='rpn_reg')])
            if self.with_cls:
                self.init_cfg['override'].append(dict(name='rpn_cls'))

    def _init_layers(self) -> None:
        """Init layers of a CascadeRPN stage."""
        adapt_cfg = copy.deepcopy(self.adapt_cfg)
        adapt_cfg['adapt_type'] = adapt_cfg.pop('type')
        self.rpn_conv = AdaptiveConv(self.in_channels, self.feat_channels,
                                     **adapt_cfg)
        if self.with_cls:
            self.rpn_cls = nn.Conv2d(self.feat_channels,
                                     self.num_anchors * self.cls_out_channels,
                                     1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward_single(self, x: Tensor, offset: Tensor) -> Tuple[Tensor]:
        """Forward function of single scale."""
        bridged_x = x
        x = self.relu(self.rpn_conv(x, offset))
        if self.bridged_feature:
            bridged_x = x  # update feature
        cls_score = self.rpn_cls(x) if self.with_cls else None
        bbox_pred = self.rpn_reg(x)
        return bridged_x, cls_score, bbox_pred

    def forward(
            self,
            feats: List[Tensor],
            offset_list: Optional[List[Tensor]] = None) -> Tuple[List[Tensor]]:
        """Forward function."""
        if offset_list is None:
            offset_list = [None for _ in range(len(feats))]
        return multi_apply(self.forward_single, feats, offset_list)

    def _region_targets_single(self, flat_anchors: Tensor, valid_flags: Tensor,
                               gt_instances: InstanceData, img_meta: dict,
                               gt_instances_ignore: InstanceData,
                               featmap_sizes: List[Tuple[int, int]],
                               num_level_anchors: List[int]) -> tuple:
        """Get anchor targets based on region for single level.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors, ).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            featmap_sizes (list[Tuple[int, int]]): Feature map size each level.
            num_level_anchors (list[int]): The number of anchors in each level.

        Returns:
            tuple:

                - labels (Tensor): Labels of each level.
                - label_weights (Tensor): Label weights of each level.
                - bbox_targets (Tensor): BBox targets of each level.
                - bbox_weights (Tensor): BBox weights of each level.
                - pos_inds (Tensor): positive samples indexes.
                - neg_inds (Tensor): negative samples indexes.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        pred_instances = InstanceData()
        pred_instances.priors = flat_anchors
        pred_instances.valid_flags = valid_flags

        assign_result = self.assigner.assign(
            pred_instances,
            gt_instances,
            img_meta,
            featmap_sizes,
            num_level_anchors,
            self.anchor_scales[0],
            self.anchor_strides,
            gt_instances_ignore=gt_instances_ignore,
            allowed_border=self.train_cfg['allowed_border'])
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_anchors = flat_anchors.shape[0]
        bbox_targets = torch.zeros_like(flat_anchors)
        bbox_weights = torch.zeros_like(flat_anchors)
        labels = flat_anchors.new_zeros(num_anchors, dtype=torch.long)
        label_weights = flat_anchors.new_zeros(num_anchors, dtype=torch.float)

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
            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def region_targets(
        self,
        anchor_list: List[List[Tensor]],
        valid_flag_list: List[List[Tensor]],
        featmap_sizes: List[Tuple[int, int]],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
        return_sampling_results: bool = False,
    ) -> tuple:
        """Compute regression and classification targets for anchors when using
        RegionAssigner.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image.
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image.
            featmap_sizes (list[Tuple[int, int]]): Feature map size each level.
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
            tuple:

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - avg_factor (int): Average factor that is used to average
                  the loss. When using sampling method, avg_factor is usually
                  the sum of positive and negative priors. When using
                  ``PseudoSampler``, ``avg_factor`` is usually equal to the
                  number of positive priors.
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = multi_apply(
             self._region_targets_single,
             concat_anchor_list,
             concat_valid_flag_list,
             batch_gt_instances,
             batch_img_metas,
             batch_gt_instances_ignore,
             featmap_sizes=featmap_sizes,
             num_level_anchors=num_level_anchors)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
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
        return res

    def get_targets(
        self,
        anchor_list: List[List[Tensor]],
        valid_flag_list: List[List[Tensor]],
        featmap_sizes: List[Tuple[int, int]],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
        return_sampling_results: bool = False,
    ) -> tuple:
        """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image.
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image.
            featmap_sizes (list[Tuple[int, int]]): Feature map size each level.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            return_sampling_results (bool): Whether to return the sampling
                results. Defaults to False.

        Returns:
            tuple:

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - avg_factor (int): Average factor that is used to average
                  the loss. When using sampling method, avg_factor is usually
                  the sum of positive and negative priors. When using
                  ``PseudoSampler``, ``avg_factor`` is usually equal to the
                  number of positive priors.
        """
        if isinstance(self.assigner, RegionAssigner):
            cls_reg_targets = self.region_targets(
                anchor_list,
                valid_flag_list,
                featmap_sizes,
                batch_gt_instances,
                batch_img_metas,
                batch_gt_instances_ignore=batch_gt_instances_ignore,
                return_sampling_results=return_sampling_results)
        else:
            cls_reg_targets = super().get_targets(
                anchor_list,
                valid_flag_list,
                batch_gt_instances,
                batch_img_metas,
                batch_gt_instances_ignore=batch_gt_instances_ignore,
                return_sampling_results=return_sampling_results)
        return cls_reg_targets

    def anchor_offset(self, anchor_list: List[List[Tensor]],
                      anchor_strides: List[int],
                      featmap_sizes: List[Tuple[int, int]]) -> List[Tensor]:
        """ Get offset for deformable conv based on anchor shape
        NOTE: currently support deformable kernel_size=3 and dilation=1

        Args:
            anchor_list (list[list[tensor])): [NI, NLVL, NA, 4] list of
                multi-level anchors
            anchor_strides (list[int]): anchor stride of each level

        Returns:
            list[tensor]: offset of DeformConv kernel with shapes of
            [NLVL, NA, 2, 18].
        """

        def _shape_offset(anchors, stride, ks=3, dilation=1):
            # currently support kernel_size=3 and dilation=1
            assert ks == 3 and dilation == 1
            pad = (ks - 1) // 2
            idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
            yy, xx = torch.meshgrid(idx, idx)  # return order matters
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            w = (anchors[:, 2] - anchors[:, 0]) / stride
            h = (anchors[:, 3] - anchors[:, 1]) / stride
            w = w / (ks - 1) - dilation
            h = h / (ks - 1) - dilation
            offset_x = w[:, None] * xx  # (NA, ks**2)
            offset_y = h[:, None] * yy  # (NA, ks**2)
            return offset_x, offset_y

        def _ctr_offset(anchors, stride, featmap_size):
            feat_h, feat_w = featmap_size
            assert len(anchors) == feat_h * feat_w

            x = (anchors[:, 0] + anchors[:, 2]) * 0.5
            y = (anchors[:, 1] + anchors[:, 3]) * 0.5
            # compute centers on feature map
            x = x / stride
            y = y / stride
            # compute predefine centers
            xx = torch.arange(0, feat_w, device=anchors.device)
            yy = torch.arange(0, feat_h, device=anchors.device)
            yy, xx = torch.meshgrid(yy, xx)
            xx = xx.reshape(-1).type_as(x)
            yy = yy.reshape(-1).type_as(y)

            offset_x = x - xx  # (NA, )
            offset_y = y - yy  # (NA, )
            return offset_x, offset_y

        num_imgs = len(anchor_list)
        num_lvls = len(anchor_list[0])
        dtype = anchor_list[0][0].dtype
        device = anchor_list[0][0].device
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        offset_list = []
        for i in range(num_imgs):
            mlvl_offset = []
            for lvl in range(num_lvls):
                c_offset_x, c_offset_y = _ctr_offset(anchor_list[i][lvl],
                                                     anchor_strides[lvl],
                                                     featmap_sizes[lvl])
                s_offset_x, s_offset_y = _shape_offset(anchor_list[i][lvl],
                                                       anchor_strides[lvl])

                # offset = ctr_offset + shape_offset
                offset_x = s_offset_x + c_offset_x[:, None]
                offset_y = s_offset_y + c_offset_y[:, None]

                # offset order (y0, x0, y1, x2, .., y8, x8, y9, x9)
                offset = torch.stack([offset_y, offset_x], dim=-1)
                offset = offset.reshape(offset.size(0), -1)  # [NA, 2*ks**2]
                mlvl_offset.append(offset)
            offset_list.append(torch.cat(mlvl_offset))  # [totalNA, 2*ks**2]
        offset_list = images_to_levels(offset_list, num_level_anchors)
        return offset_list

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            anchors: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor, avg_factor: int) -> tuple:
        """Loss function on single scale."""
        # classification loss
        if self.with_cls:
            labels = labels.reshape(-1)
            label_weights = label_weights.reshape(-1)
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(-1, self.cls_out_channels)
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=avg_factor)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_reg = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        if self.with_cls:
            return loss_cls, loss_reg
        return None, loss_reg

    def loss_by_feat(
        self,
        anchor_list: List[List[Tensor]],
        valid_flag_list: List[List[Tensor]],
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Compute losses of the head.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image.
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
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
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            featmap_sizes,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            return_sampling_results=True)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor, sampling_results_list) = cls_reg_targets
        if not sampling_results_list[0].avg_factor_with_neg:
            # 200 is hard-coded average factor,
            # which follows guided anchoring.
            avg_factor = sum([label.numel() for label in labels_list]) / 200.0

        # change per image, per level anchor_list to per_level, per_image
        mlvl_anchor_list = list(zip(*anchor_list))
        # concat mlvl_anchor_list
        mlvl_anchor_list = [
            torch.cat(anchors, dim=0) for anchors in mlvl_anchor_list
        ]

        losses = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            mlvl_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor)
        if self.with_cls:
            return dict(loss_rpn_cls=losses[0], loss_rpn_reg=losses[1])
        return dict(loss_rpn_reg=losses[1])

    def predict_by_feat(self,
                        anchor_list: List[List[Tensor]],
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        batch_img_metas: List[dict],
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False) -> InstanceList:
        """Get proposal predict. Overriding to enable input ``anchor_list``
        from outside.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image.
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            batch_img_metas (list[dict], Optional): Image meta info.
            cfg (:obj:`ConfigDict`, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            proposals = self._predict_by_feat_single(
                cls_scores=cls_score_list,
                bbox_preds=bbox_pred_list,
                mlvl_anchors=anchor_list[img_id],
                img_meta=batch_img_metas[img_id],
                cfg=cfg,
                rescale=rescale)
            result_list.append(proposals)
        return result_list

    def _predict_by_feat_single(self,
                                cls_scores: List[Tensor],
                                bbox_preds: List[Tensor],
                                mlvl_anchors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False) -> InstanceData:
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has
                shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference from all scale
                levels of a single image, each item has shape
                (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (:obj:`ConfigDict`): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]

            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        bboxes = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_meta['img_shape'])

        proposals = InstanceData()
        proposals.bboxes = bboxes
        proposals.scores = torch.cat(mlvl_scores)
        proposals.level_ids = torch.cat(level_ids)

        return self._bbox_post_process(
            results=proposals, cfg=cfg, rescale=rescale, img_meta=img_meta)

    def refine_bboxes(self, anchor_list: List[List[Tensor]],
                      bbox_preds: List[Tensor],
                      img_metas: List[dict]) -> List[List[Tensor]]:
        """Refine bboxes through stages."""
        num_levels = len(bbox_preds)
        new_anchor_list = []
        for img_id in range(len(img_metas)):
            mlvl_anchors = []
            for i in range(num_levels):
                bbox_pred = bbox_preds[i][img_id].detach()
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                img_shape = img_metas[img_id]['img_shape']
                bboxes = self.bbox_coder.decode(anchor_list[img_id][i],
                                                bbox_pred, img_shape)
                mlvl_anchors.append(bboxes)
            new_anchor_list.append(mlvl_anchors)
        return new_anchor_list

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, _, batch_img_metas = outputs

        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        device = x[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        if self.adapt_cfg['type'] == 'offset':
            offset_list = self.anchor_offset(anchor_list, self.anchor_strides,
                                             featmap_sizes)
        else:
            offset_list = None

        x, cls_score, bbox_pred = self(x, offset_list)
        rpn_loss_inputs = (anchor_list, valid_flag_list, cls_score, bbox_pred,
                           batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*rpn_loss_inputs)

        return losses

    def loss_and_predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None,
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.
            proposal_cfg (:obj`ConfigDict`, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image after the post process.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, _, batch_img_metas = outputs

        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        device = x[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        if self.adapt_cfg['type'] == 'offset':
            offset_list = self.anchor_offset(anchor_list, self.anchor_strides,
                                             featmap_sizes)
        else:
            offset_list = None

        x, cls_score, bbox_pred = self(x, offset_list)
        rpn_loss_inputs = (anchor_list, valid_flag_list, cls_score, bbox_pred,
                           batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*rpn_loss_inputs)

        predictions = self.predict_by_feat(
            anchor_list,
            cls_score,
            bbox_pred,
            batch_img_metas=batch_img_metas,
            cfg=proposal_cfg)
        return losses, predictions

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        device = x[0].device
        anchor_list, _ = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        if self.adapt_cfg['type'] == 'offset':
            offset_list = self.anchor_offset(anchor_list, self.anchor_strides,
                                             featmap_sizes)
        else:
            offset_list = None

        x, cls_score, bbox_pred = self(x, offset_list)
        predictions = self.stages[-1].predict_by_feat(
            anchor_list,
            cls_score,
            bbox_pred,
            batch_img_metas=batch_img_metas,
            rescale=rescale)
        return predictions


@MODELS.register_module()
class CascadeRPNHead(BaseDenseHead):
    """The CascadeRPNHead will predict more accurate region proposals, which is
    required for two-stage detectors (such as Fast/Faster R-CNN). CascadeRPN
    consists of a sequence of RPNStage to progressively improve the accuracy of
    the detected proposals.

    More details can be found in ``https://arxiv.org/abs/1909.06720``.

    Args:
        num_stages (int): number of CascadeRPN stages.
        stages (list[:obj:`ConfigDict` or dict]): list of configs to build
            the stages.
        train_cfg (list[:obj:`ConfigDict` or dict]): list of configs at
            training time each stage.
        test_cfg (:obj:`ConfigDict` or dict): config at testing time.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or \
            list[dict]): Initialization config dict.
    """

    def __init__(self,
                 num_classes: int,
                 num_stages: int,
                 stages: List[ConfigType],
                 train_cfg: List[ConfigType],
                 test_cfg: ConfigType,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert num_classes == 1, 'Only support num_classes == 1'
        assert num_stages == len(stages)
        self.num_stages = num_stages
        # Be careful! Pretrained weights cannot be loaded when use
        # nn.ModuleList
        self.stages = ModuleList()
        for i in range(len(stages)):
            train_cfg_i = train_cfg[i] if train_cfg is not None else None
            stages[i].update(train_cfg=train_cfg_i)
            stages[i].update(test_cfg=test_cfg)
            self.stages.append(MODELS.build(stages[i]))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss_by_feat(self):
        """loss_by_feat() is implemented in StageCascadeRPNHead."""
        pass

    def predict_by_feat(self):
        """predict_by_feat() is implemented in StageCascadeRPNHead."""
        pass

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, _, batch_img_metas = outputs

        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        device = x[0].device
        anchor_list, valid_flag_list = self.stages[0].get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        losses = dict()

        for i in range(self.num_stages):
            stage = self.stages[i]

            if stage.adapt_cfg['type'] == 'offset':
                offset_list = stage.anchor_offset(anchor_list,
                                                  stage.anchor_strides,
                                                  featmap_sizes)
            else:
                offset_list = None
            x, cls_score, bbox_pred = stage(x, offset_list)
            rpn_loss_inputs = (anchor_list, valid_flag_list, cls_score,
                               bbox_pred, batch_gt_instances, batch_img_metas)
            stage_loss = stage.loss_by_feat(*rpn_loss_inputs)
            for name, value in stage_loss.items():
                losses['s{}.{}'.format(i, name)] = value

            # refine boxes
            if i < self.num_stages - 1:
                anchor_list = stage.refine_bboxes(anchor_list, bbox_pred,
                                                  batch_img_metas)

        return losses

    def loss_and_predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None,
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.
            proposal_cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image after the post process.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, _, batch_img_metas = outputs

        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        device = x[0].device
        anchor_list, valid_flag_list = self.stages[0].get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        losses = dict()

        for i in range(self.num_stages):
            stage = self.stages[i]

            if stage.adapt_cfg['type'] == 'offset':
                offset_list = stage.anchor_offset(anchor_list,
                                                  stage.anchor_strides,
                                                  featmap_sizes)
            else:
                offset_list = None
            x, cls_score, bbox_pred = stage(x, offset_list)
            rpn_loss_inputs = (anchor_list, valid_flag_list, cls_score,
                               bbox_pred, batch_gt_instances, batch_img_metas)
            stage_loss = stage.loss_by_feat(*rpn_loss_inputs)
            for name, value in stage_loss.items():
                losses['s{}.{}'.format(i, name)] = value

            # refine boxes
            if i < self.num_stages - 1:
                anchor_list = stage.refine_bboxes(anchor_list, bbox_pred,
                                                  batch_img_metas)

        predictions = self.stages[-1].predict_by_feat(
            anchor_list,
            cls_score,
            bbox_pred,
            batch_img_metas=batch_img_metas,
            cfg=proposal_cfg)
        return losses, predictions

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        device = x[0].device
        anchor_list, _ = self.stages[0].get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        for i in range(self.num_stages):
            stage = self.stages[i]
            if stage.adapt_cfg['type'] == 'offset':
                offset_list = stage.anchor_offset(anchor_list,
                                                  stage.anchor_strides,
                                                  featmap_sizes)
            else:
                offset_list = None
            x, cls_score, bbox_pred = stage(x, offset_list)
            if i < self.num_stages - 1:
                anchor_list = stage.refine_bboxes(anchor_list, bbox_pred,
                                                  batch_img_metas)

        predictions = self.stages[-1].predict_by_feat(
            anchor_list,
            cls_score,
            bbox_pred,
            batch_img_metas=batch_img_metas,
            rescale=rescale)
        return predictions
