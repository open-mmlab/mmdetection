# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d, MaskedConv2d
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList)
from ..layers import multiclass_nms
from ..task_modules.prior_generators import anchor_inside_flags, calc_region
from ..task_modules.samplers import PseudoSampler
from ..utils import images_to_levels, multi_apply, unmap
from .anchor_head import AnchorHead


class FeatureAdaption(BaseModule):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deform conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size. Defaults to 3.
        deform_groups (int): Deformable conv group size. Defaults to 4.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or \
            list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        deform_groups: int = 4,
        init_cfg: MultiConfig = dict(
            type='Normal',
            layer='Conv2d',
            std=0.1,
            override=dict(type='Normal', name='conv_adaption', std=0.01))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            2, deform_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, shape: Tensor) -> Tensor:
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x


@MODELS.register_module()
class GuidedAnchorHead(AnchorHead):
    """Guided-Anchor-based head (GA-RPN, GA-RetinaNet, etc.).

    This GuidedAnchorHead will predict high-quality feature guided
    anchors and locations where anchors will be kept in inference.
    There are mainly 3 categories of bounding-boxes.

    - Sampled 9 pairs for target assignment. (approxes)
    - The square boxes where the predicted anchors are based on. (squares)
    - Guided anchors.

    Please refer to https://arxiv.org/abs/1901.03278 for more details.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Defaults to 256.
        approx_anchor_generator (:obj:`ConfigDict` or dict): Config dict
            for approx generator
        square_anchor_generator (:obj:`ConfigDict` or dict): Config dict
            for square generator
        anchor_coder (:obj:`ConfigDict` or dict): Config dict for anchor coder
        bbox_coder (:obj:`ConfigDict` or dict): Config dict for bbox coder
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Defaults to False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        deform_groups: (int): Group number of DCN in FeatureAdaption module.
            Defaults to 4.
        loc_filter_thr (float): Threshold to filter out unconcerned regions.
            Defaults to 0.01.
        loss_loc (:obj:`ConfigDict` or dict): Config of location loss.
        loss_shape (:obj:`ConfigDict` or dict): Config of anchor shape loss.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of bbox regression loss.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or \
            list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        approx_anchor_generator: ConfigType = dict(
            type='AnchorGenerator',
            octave_base_scale=8,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        square_anchor_generator: ConfigType = dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[8],
            strides=[4, 8, 16, 32, 64]),
        anchor_coder: ConfigType = dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        bbox_coder: ConfigType = dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        reg_decoded_bbox: bool = False,
        deform_groups: int = 4,
        loc_filter_thr: float = 0.01,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        loss_loc: ConfigType = dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape: ConfigType = dict(
            type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls: ConfigType = dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox: ConfigType = dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
        init_cfg: MultiConfig = dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal', name='conv_loc', std=0.01, lbias_prob=0.01))
    ) -> None:
        super(AnchorHead, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.deform_groups = deform_groups
        self.loc_filter_thr = loc_filter_thr

        # build approx_anchor_generator and square_anchor_generator
        assert (approx_anchor_generator['octave_base_scale'] ==
                square_anchor_generator['scales'][0])
        assert (approx_anchor_generator['strides'] ==
                square_anchor_generator['strides'])
        self.approx_anchor_generator = TASK_UTILS.build(
            approx_anchor_generator)
        self.square_anchor_generator = TASK_UTILS.build(
            square_anchor_generator)
        self.approxs_per_octave = self.approx_anchor_generator \
            .num_base_priors[0]

        self.reg_decoded_bbox = reg_decoded_bbox

        # one anchor per location
        self.num_base_priors = self.square_anchor_generator.num_base_priors[0]

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.loc_focal_loss = loss_loc['type'] in ['FocalLoss']
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

        # build bbox_coder
        self.anchor_coder = TASK_UTILS.build(anchor_coder)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)

        # build losses
        self.loss_loc = MODELS.build(loss_loc)
        self.loss_shape = MODELS.build(loss_shape)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            # use PseudoSampler when no sampler in train_cfg
            if train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'], default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler()

            self.ga_assigner = TASK_UTILS.build(self.train_cfg['ga_assigner'])
            if train_cfg.get('ga_sampler', None) is not None:
                self.ga_sampler = TASK_UTILS.build(
                    self.train_cfg['ga_sampler'],
                    default_args=dict(context=self))
            else:
                self.ga_sampler = PseudoSampler()

        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.conv_loc = nn.Conv2d(self.in_channels, 1, 1)
        self.conv_shape = nn.Conv2d(self.in_channels, self.num_base_priors * 2,
                                    1)
        self.feature_adaption = FeatureAdaption(
            self.in_channels,
            self.feat_channels,
            kernel_size=3,
            deform_groups=self.deform_groups)
        self.conv_cls = MaskedConv2d(
            self.feat_channels, self.num_base_priors * self.cls_out_channels,
            1)
        self.conv_reg = MaskedConv2d(self.feat_channels,
                                     self.num_base_priors * 4, 1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor]:
        """Forward feature of a single scale level."""
        loc_pred = self.conv_loc(x)
        shape_pred = self.conv_shape(x)
        x = self.feature_adaption(x, shape_pred)
        # masked conv is only used during inference for speed-up
        if not self.training:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None
        cls_score = self.conv_cls(x, mask)
        bbox_pred = self.conv_reg(x, mask)
        return cls_score, bbox_pred, shape_pred, loc_pred

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network."""
        return multi_apply(self.forward_single, x)

    def get_sampled_approxs(self,
                            featmap_sizes: List[Tuple[int, int]],
                            batch_img_metas: List[dict],
                            device: str = 'cuda') -> tuple:
        """Get sampled approxs and inside flags according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            batch_img_metas (list[dict]): Image meta info.
            device (str): device for returned tensors

        Returns:
            tuple: approxes of each image, inside flags of each image
        """
        num_imgs = len(batch_img_metas)

        # since feature map sizes of all images are the same, we only compute
        # approxes for one time
        multi_level_approxs = self.approx_anchor_generator.grid_priors(
            featmap_sizes, device=device)
        approxs_list = [multi_level_approxs for _ in range(num_imgs)]

        # for each image, we compute inside flags of multi level approxes
        inside_flag_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_flags = []
            multi_level_approxs = approxs_list[img_id]

            # obtain valid flags for each approx first
            multi_level_approx_flags = self.approx_anchor_generator \
                .valid_flags(featmap_sizes,
                             img_meta['pad_shape'],
                             device=device)

            for i, flags in enumerate(multi_level_approx_flags):
                approxs = multi_level_approxs[i]
                inside_flags_list = []
                for j in range(self.approxs_per_octave):
                    split_valid_flags = flags[j::self.approxs_per_octave]
                    split_approxs = approxs[j::self.approxs_per_octave, :]
                    inside_flags = anchor_inside_flags(
                        split_approxs, split_valid_flags,
                        img_meta['img_shape'][:2],
                        self.train_cfg['allowed_border'])
                    inside_flags_list.append(inside_flags)
                # inside_flag for a position is true if any anchor in this
                # position is true
                inside_flags = (
                    torch.stack(inside_flags_list, 0).sum(dim=0) > 0)
                multi_level_flags.append(inside_flags)
            inside_flag_list.append(multi_level_flags)
        return approxs_list, inside_flag_list

    def get_anchors(self,
                    featmap_sizes: List[Tuple[int, int]],
                    shape_preds: List[Tensor],
                    loc_preds: List[Tensor],
                    batch_img_metas: List[dict],
                    use_loc_filter: bool = False,
                    device: str = 'cuda') -> tuple:
        """Get squares according to feature map sizes and guided anchors.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            shape_preds (list[tensor]): Multi-level shape predictions.
            loc_preds (list[tensor]): Multi-level location predictions.
            batch_img_metas (list[dict]): Image meta info.
            use_loc_filter (bool): Use loc filter or not. Defaults to False
            device (str): device for returned tensors.
                Defaults to `cuda`.

        Returns:
            tuple: square approxs of each image, guided anchors of each image,
            loc masks of each image.
        """
        num_imgs = len(batch_img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # squares for one time
        multi_level_squares = self.square_anchor_generator.grid_priors(
            featmap_sizes, device=device)
        squares_list = [multi_level_squares for _ in range(num_imgs)]

        # for each image, we compute multi level guided anchors
        guided_anchors_list = []
        loc_mask_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_guided_anchors = []
            multi_level_loc_mask = []
            for i in range(num_levels):
                squares = squares_list[img_id][i]
                shape_pred = shape_preds[i][img_id]
                loc_pred = loc_preds[i][img_id]
                guided_anchors, loc_mask = self._get_guided_anchors_single(
                    squares,
                    shape_pred,
                    loc_pred,
                    use_loc_filter=use_loc_filter)
                multi_level_guided_anchors.append(guided_anchors)
                multi_level_loc_mask.append(loc_mask)
            guided_anchors_list.append(multi_level_guided_anchors)
            loc_mask_list.append(multi_level_loc_mask)
        return squares_list, guided_anchors_list, loc_mask_list

    def _get_guided_anchors_single(
            self,
            squares: Tensor,
            shape_pred: Tensor,
            loc_pred: Tensor,
            use_loc_filter: bool = False) -> Tuple[Tensor]:
        """Get guided anchors and loc masks for a single level.

        Args:
            squares (tensor): Squares of a single level.
            shape_pred (tensor): Shape predictions of a single level.
            loc_pred (tensor): Loc predictions of a single level.
            use_loc_filter (list[tensor]): Use loc filter or not.
                Defaults to False.

        Returns:
            tuple: guided anchors, location masks
        """
        # calculate location filtering mask
        loc_pred = loc_pred.sigmoid().detach()
        if use_loc_filter:
            loc_mask = loc_pred >= self.loc_filter_thr
        else:
            loc_mask = loc_pred >= 0.0
        mask = loc_mask.permute(1, 2, 0).expand(-1, -1, self.num_base_priors)
        mask = mask.contiguous().view(-1)
        # calculate guided anchors
        squares = squares[mask]
        anchor_deltas = shape_pred.permute(1, 2, 0).contiguous().view(
            -1, 2).detach()[mask]
        bbox_deltas = anchor_deltas.new_full(squares.size(), 0)
        bbox_deltas[:, 2:] = anchor_deltas
        guided_anchors = self.anchor_coder.decode(
            squares, bbox_deltas, wh_ratio_clip=1e-6)
        return guided_anchors, mask

    def ga_loc_targets(self, batch_gt_instances: InstanceList,
                       featmap_sizes: List[Tuple[int, int]]) -> tuple:
        """Compute location targets for guided anchoring.

        Each feature map is divided into positive, negative and ignore regions.
        - positive regions: target 1, weight 1
        - ignore regions: target 0, weight 0
        - negative regions: target 0, weight 0.1

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            featmap_sizes (list[tuple]): Multi level sizes of each feature
                maps.

        Returns:
            tuple: Returns a tuple containing location targets.
        """
        anchor_scale = self.approx_anchor_generator.octave_base_scale
        anchor_strides = self.approx_anchor_generator.strides
        # Currently only supports same stride in x and y direction.
        for stride in anchor_strides:
            assert (stride[0] == stride[1])
        anchor_strides = [stride[0] for stride in anchor_strides]

        center_ratio = self.train_cfg['center_ratio']
        ignore_ratio = self.train_cfg['ignore_ratio']
        img_per_gpu = len(batch_gt_instances)
        num_lvls = len(featmap_sizes)
        r1 = (1 - center_ratio) / 2
        r2 = (1 - ignore_ratio) / 2
        all_loc_targets = []
        all_loc_weights = []
        all_ignore_map = []
        for lvl_id in range(num_lvls):
            h, w = featmap_sizes[lvl_id]
            loc_targets = torch.zeros(
                img_per_gpu,
                1,
                h,
                w,
                device=batch_gt_instances[0].bboxes.device,
                dtype=torch.float32)
            loc_weights = torch.full_like(loc_targets, -1)
            ignore_map = torch.zeros_like(loc_targets)
            all_loc_targets.append(loc_targets)
            all_loc_weights.append(loc_weights)
            all_ignore_map.append(ignore_map)
        for img_id in range(img_per_gpu):
            gt_bboxes = batch_gt_instances[img_id].bboxes
            scale = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) *
                               (gt_bboxes[:, 3] - gt_bboxes[:, 1]))
            min_anchor_size = scale.new_full(
                (1, ), float(anchor_scale * anchor_strides[0]))
            # assign gt bboxes to different feature levels w.r.t. their scales
            target_lvls = torch.floor(
                torch.log2(scale) - torch.log2(min_anchor_size) + 0.5)
            target_lvls = target_lvls.clamp(min=0, max=num_lvls - 1).long()
            for gt_id in range(gt_bboxes.size(0)):
                lvl = target_lvls[gt_id].item()
                # rescaled to corresponding feature map
                gt_ = gt_bboxes[gt_id, :4] / anchor_strides[lvl]
                # calculate ignore regions
                ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(
                    gt_, r2, featmap_sizes[lvl])
                # calculate positive (center) regions
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = calc_region(
                    gt_, r1, featmap_sizes[lvl])
                all_loc_targets[lvl][img_id, 0, ctr_y1:ctr_y2 + 1,
                                     ctr_x1:ctr_x2 + 1] = 1
                all_loc_weights[lvl][img_id, 0, ignore_y1:ignore_y2 + 1,
                                     ignore_x1:ignore_x2 + 1] = 0
                all_loc_weights[lvl][img_id, 0, ctr_y1:ctr_y2 + 1,
                                     ctr_x1:ctr_x2 + 1] = 1
                # calculate ignore map on nearby low level feature
                if lvl > 0:
                    d_lvl = lvl - 1
                    # rescaled to corresponding feature map
                    gt_ = gt_bboxes[gt_id, :4] / anchor_strides[d_lvl]
                    ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(
                        gt_, r2, featmap_sizes[d_lvl])
                    all_ignore_map[d_lvl][img_id, 0, ignore_y1:ignore_y2 + 1,
                                          ignore_x1:ignore_x2 + 1] = 1
                # calculate ignore map on nearby high level feature
                if lvl < num_lvls - 1:
                    u_lvl = lvl + 1
                    # rescaled to corresponding feature map
                    gt_ = gt_bboxes[gt_id, :4] / anchor_strides[u_lvl]
                    ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(
                        gt_, r2, featmap_sizes[u_lvl])
                    all_ignore_map[u_lvl][img_id, 0, ignore_y1:ignore_y2 + 1,
                                          ignore_x1:ignore_x2 + 1] = 1
        for lvl_id in range(num_lvls):
            # ignore negative regions w.r.t. ignore map
            all_loc_weights[lvl_id][(all_loc_weights[lvl_id] < 0)
                                    & (all_ignore_map[lvl_id] > 0)] = 0
            # set negative regions with weight 0.1
            all_loc_weights[lvl_id][all_loc_weights[lvl_id] < 0] = 0.1
        # loc average factor to balance loss
        loc_avg_factor = sum(
            [t.size(0) * t.size(-1) * t.size(-2)
             for t in all_loc_targets]) / 200
        return all_loc_targets, all_loc_weights, loc_avg_factor

    def _ga_shape_target_single(self,
                                flat_approxs: Tensor,
                                inside_flags: Tensor,
                                flat_squares: Tensor,
                                gt_instances: InstanceData,
                                gt_instances_ignore: Optional[InstanceData],
                                img_meta: dict,
                                unmap_outputs: bool = True) -> tuple:
        """Compute guided anchoring targets.

        This function returns sampled anchors and gt bboxes directly
        rather than calculates regression targets.

        Args:
            flat_approxs (Tensor): flat approxs of a single image,
                shape (n, 4)
            inside_flags (Tensor): inside flags of a single image,
                shape (n, ).
            flat_squares (Tensor): flat squares of a single image,
                shape (approxs_per_octave * n, 4)
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
            img_meta (dict): Meta info of a single image.
            unmap_outputs (bool): unmap outputs or not.

        Returns:
            tuple: Returns a tuple containing shape targets of each image.
        """
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        num_square = flat_squares.size(0)
        approxs = flat_approxs.view(num_square, self.approxs_per_octave, 4)
        approxs = approxs[inside_flags, ...]
        squares = flat_squares[inside_flags, :]

        pred_instances = InstanceData()
        pred_instances.priors = squares
        pred_instances.approxs = approxs

        assign_result = self.ga_assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            gt_instances_ignore=gt_instances_ignore)
        sampling_result = self.ga_sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)

        bbox_anchors = torch.zeros_like(squares)
        bbox_gts = torch.zeros_like(squares)
        bbox_weights = torch.zeros_like(squares)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            bbox_anchors[pos_inds, :] = sampling_result.pos_bboxes
            bbox_gts[pos_inds, :] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds, :] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_squares.size(0)
            bbox_anchors = unmap(bbox_anchors, num_total_anchors, inside_flags)
            bbox_gts = unmap(bbox_gts, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (bbox_anchors, bbox_gts, bbox_weights, pos_inds, neg_inds,
                sampling_result)

    def ga_shape_targets(self,
                         approx_list: List[List[Tensor]],
                         inside_flag_list: List[List[Tensor]],
                         square_list: List[List[Tensor]],
                         batch_gt_instances: InstanceList,
                         batch_img_metas: List[dict],
                         batch_gt_instances_ignore: OptInstanceList = None,
                         unmap_outputs: bool = True) -> tuple:
        """Compute guided anchoring targets.

        Args:
            approx_list (list[list[Tensor]]): Multi level approxs of each
                image.
            inside_flag_list (list[list[Tensor]]): Multi level inside flags
                of each image.
            square_list (list[list[Tensor]]): Multi level squares of each
                image.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): unmap outputs or not. Defaults to None.

        Returns:
            tuple:  Returns a tuple containing shape targets.
        """
        num_imgs = len(batch_img_metas)
        assert len(approx_list) == len(inside_flag_list) == len(
            square_list) == num_imgs
        # anchor number of multi levels
        num_level_squares = [squares.size(0) for squares in square_list[0]]
        # concat all level anchors and flags to a single tensor
        inside_flag_flat_list = []
        approx_flat_list = []
        square_flat_list = []
        for i in range(num_imgs):
            assert len(square_list[i]) == len(inside_flag_list[i])
            inside_flag_flat_list.append(torch.cat(inside_flag_list[i]))
            approx_flat_list.append(torch.cat(approx_list[i]))
            square_flat_list.append(torch.cat(square_list[i]))

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None for _ in range(num_imgs)]
        (all_bbox_anchors, all_bbox_gts, all_bbox_weights, pos_inds_list,
         neg_inds_list, sampling_results_list) = multi_apply(
             self._ga_shape_target_single,
             approx_flat_list,
             inside_flag_flat_list,
             square_flat_list,
             batch_gt_instances,
             batch_gt_instances_ignore,
             batch_img_metas,
             unmap_outputs=unmap_outputs)
        # sampled anchors of all images
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        # split targets to a list w.r.t. multiple levels
        bbox_anchors_list = images_to_levels(all_bbox_anchors,
                                             num_level_squares)
        bbox_gts_list = images_to_levels(all_bbox_gts, num_level_squares)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_squares)
        return (bbox_anchors_list, bbox_gts_list, bbox_weights_list,
                avg_factor)

    def loss_shape_single(self, shape_pred: Tensor, bbox_anchors: Tensor,
                          bbox_gts: Tensor, anchor_weights: Tensor,
                          avg_factor: int) -> Tensor:
        """Compute shape loss in single level."""
        shape_pred = shape_pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        bbox_anchors = bbox_anchors.contiguous().view(-1, 4)
        bbox_gts = bbox_gts.contiguous().view(-1, 4)
        anchor_weights = anchor_weights.contiguous().view(-1, 4)
        bbox_deltas = bbox_anchors.new_full(bbox_anchors.size(), 0)
        bbox_deltas[:, 2:] += shape_pred
        # filter out negative samples to speed-up weighted_bounded_iou_loss
        inds = torch.nonzero(
            anchor_weights[:, 0] > 0, as_tuple=False).squeeze(1)
        bbox_deltas_ = bbox_deltas[inds]
        bbox_anchors_ = bbox_anchors[inds]
        bbox_gts_ = bbox_gts[inds]
        anchor_weights_ = anchor_weights[inds]
        pred_anchors_ = self.anchor_coder.decode(
            bbox_anchors_, bbox_deltas_, wh_ratio_clip=1e-6)
        loss_shape = self.loss_shape(
            pred_anchors_, bbox_gts_, anchor_weights_, avg_factor=avg_factor)
        return loss_shape

    def loss_loc_single(self, loc_pred: Tensor, loc_target: Tensor,
                        loc_weight: Tensor, avg_factor: float) -> Tensor:
        """Compute location loss in single level."""
        loss_loc = self.loss_loc(
            loc_pred.reshape(-1, 1),
            loc_target.reshape(-1).long(),
            loc_weight.reshape(-1),
            avg_factor=avg_factor)
        return loss_loc

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            shape_preds: List[Tensor],
            loc_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            shape_preds (list[Tensor]): shape predictions for each scale
                level with shape (N, 1, H, W).
            loc_preds (list[Tensor]): location predictions for each scale
                level with shape (N, num_anchors * 2, H, W).
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

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.approx_anchor_generator.num_levels

        device = cls_scores[0].device

        # get loc targets
        loc_targets, loc_weights, loc_avg_factor = self.ga_loc_targets(
            batch_gt_instances, featmap_sizes)

        # get sampled approxes
        approxs_list, inside_flag_list = self.get_sampled_approxs(
            featmap_sizes, batch_img_metas, device=device)
        # get squares and guided anchors
        squares_list, guided_anchors_list, _ = self.get_anchors(
            featmap_sizes,
            shape_preds,
            loc_preds,
            batch_img_metas,
            device=device)

        # get shape targets
        shape_targets = self.ga_shape_targets(approxs_list, inside_flag_list,
                                              squares_list, batch_gt_instances,
                                              batch_img_metas)
        (bbox_anchors_list, bbox_gts_list, anchor_weights_list,
         ga_avg_factor) = shape_targets

        # get anchor targets
        cls_reg_targets = self.get_targets(
            guided_anchors_list,
            inside_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor) = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [
            anchors.size(0) for anchors in guided_anchors_list[0]
        ]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        for i in range(len(guided_anchors_list)):
            concat_anchor_list.append(torch.cat(guided_anchors_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        # get classification and bbox regression losses
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

        # get anchor location loss
        losses_loc = []
        for i in range(len(loc_preds)):
            loss_loc = self.loss_loc_single(
                loc_preds[i],
                loc_targets[i],
                loc_weights[i],
                avg_factor=loc_avg_factor)
            losses_loc.append(loss_loc)

        # get anchor shape loss
        losses_shape = []
        for i in range(len(shape_preds)):
            loss_shape = self.loss_shape_single(
                shape_preds[i],
                bbox_anchors_list[i],
                bbox_gts_list[i],
                anchor_weights_list[i],
                avg_factor=ga_avg_factor)
            losses_shape.append(loss_shape)

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_shape=losses_shape,
            loss_loc=losses_loc)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        shape_preds: List[Tensor],
                        loc_preds: List[Tensor],
                        batch_img_metas: List[dict],
                        cfg: OptConfigType = None,
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            shape_preds (list[Tensor]): shape predictions for each scale
                level with shape (N, 1, H, W).
            loc_preds (list[Tensor]): location predictions for each scale
                level with shape (N, num_anchors * 2, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4), the last
              dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds) == len(shape_preds) == len(
            loc_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        # get guided anchors
        _, guided_anchors, loc_masks = self.get_anchors(
            featmap_sizes,
            shape_preds,
            loc_preds,
            batch_img_metas,
            use_loc_filter=not self.training,
            device=device)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            guided_anchor_list = [
                guided_anchors[img_id][i].detach() for i in range(num_levels)
            ]
            loc_mask_list = [
                loc_masks[img_id][i].detach() for i in range(num_levels)
            ]
            proposals = self._predict_by_feat_single(
                cls_scores=cls_score_list,
                bbox_preds=bbox_pred_list,
                mlvl_anchors=guided_anchor_list,
                mlvl_masks=loc_mask_list,
                img_meta=batch_img_metas[img_id],
                cfg=cfg,
                rescale=rescale)
            result_list.append(proposals)
        return result_list

    def _predict_by_feat_single(self,
                                cls_scores: List[Tensor],
                                bbox_preds: List[Tensor],
                                mlvl_anchors: List[Tensor],
                                mlvl_masks: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigType,
                                rescale: bool = False) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            mlvl_anchors (list[Tensor]): Each element in the list is
                the anchors of a single level in feature pyramid. it has
                shape (num_priors, 4).
            mlvl_masks (list[Tensor]): Each element in the list is location
                masks of a single level.
            img_meta (dict): Image meta info.
            cfg (:obj:`ConfigDict` or dict): Test / postprocessing
                configuration, if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4), the last
              dimension 4 arrange as (x1, y1, x2, y2).
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors, mask in zip(cls_scores, bbox_preds,
                                                       mlvl_anchors,
                                                       mlvl_masks):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # if no location is kept, end.
            if mask.sum() == 0:
                continue
            # reshape scores and bbox_pred
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # filter scores, bbox_pred w.r.t. mask.
            # anchors are filtered in get_anchors() beforehand.
            scores = scores[mask, :]
            bbox_pred = bbox_pred[mask, :]
            if scores.dim() == 0:
                anchors = anchors.unsqueeze(0)
                scores = scores.unsqueeze(0)
                bbox_pred = bbox_pred.unsqueeze(0)
            # filter anchors, bbox_pred, scores w.r.t. scores
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_anchors.append(anchors)
            mlvl_scores.append(scores)

        mlvl_bbox_preds = torch.cat(mlvl_bbox_preds)
        mlvl_anchors = torch.cat(mlvl_valid_anchors)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_bboxes = self.bbox_coder.decode(
            mlvl_anchors, mlvl_bbox_preds, max_shape=img_meta['img_shape'])

        if rescale:
            assert img_meta.get('scale_factor') is not None
            mlvl_bboxes /= mlvl_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        # multi class NMS
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)

        results = InstanceData()
        results.bboxes = det_bboxes[:, :-1]
        results.scores = det_bboxes[:, -1]
        results.labels = det_labels
        return results
