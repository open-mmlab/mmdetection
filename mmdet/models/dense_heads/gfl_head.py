# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList, reduce_mean)
from ..task_modules.prior_generators import anchor_inside_flags
from ..task_modules.samplers import PseudoSampler
from ..utils import (filter_scores_and_topk, images_to_levels, multi_apply,
                     unmap)
from .anchor_head import AnchorHead


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: ``sum{P(y_i) * y_i}``,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Defaults to 16.
            You may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


@MODELS.register_module()
class GFLHead(AnchorHead):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    GFL head structure is similar with ATSS, however GFL uses
    1) joint representation for classification and localization quality, and
    2) flexible General distribution for bounding box locations,
    which are supervised by
    Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

    https://arxiv.org/abs/2006.04388

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Defaults to 4.
        conv_cfg (:obj:`ConfigDict` or dict, optional): dictionary to construct
            and config conv layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer. Default: dict(type='GN', num_groups=32,
            requires_grad=True).
        loss_qfl (:obj:`ConfigDict` or dict): Config of Quality Focal Loss
            (QFL).
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder. Defaults
             to 'DistancePointBBoxCoder'.
        reg_max (int): Max value of integral set :math: ``{0, ..., reg_max}``
            in QFL setting. Defaults to 16.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    Example:
        >>> self = GFLHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 stacked_convs: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 loss_dfl: ConfigType = dict(
                     type='DistributionFocalLoss', loss_weight=0.25),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 reg_max: int = 16,
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='gfl_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            bbox_coder=bbox_coder,
            init_cfg=init_cfg,
            **kwargs)

        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            if self.train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'], default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler(context=self)

        self.integral = Integral(self.reg_max)
        self.loss_dfl = MODELS.build(loss_dfl)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        assert self.num_anchors == 1, 'anchor free version'
        self.gfl_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction

            - cls_scores (list[Tensor]): Classification and quality (IoU)
              joint scores for all scale levels, each is a 4D-tensor,
              the channel number is num_classes.
            - bbox_preds (list[Tensor]): Box distribution logits for all
              scale levels, each is a 4D-tensor, the channel number is
              4*(n+1), n is max value of integral set.
        """
        return multi_apply(self.forward_single, x, self.scales)

    def forward_single(self, x: Tensor, scale: Scale) -> Sequence[Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:

            - cls_score (Tensor): Cls and quality joint scores for a single
              scale level the channel number is num_classes.
            - bbox_pred (Tensor): Box distribution logits for a single scale
              level, the channel number is 4*(n+1), n is max value of
              integral set.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.gfl_cls(cls_feat)
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        return cls_score, bbox_pred

    def anchor_center(self, anchors: Tensor) -> Tensor:
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), ``xyxy`` format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), ``xy`` format.
        """
        anchors_cx = (anchors[..., 2] + anchors[..., 0]) / 2
        anchors_cy = (anchors[..., 3] + anchors[..., 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_by_feat_single(self, anchors: Tensor, cls_score: Tensor,
                            bbox_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            stride: Tuple[int], avg_factor: int) -> dict:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            stride (Tuple[int]): Stride in this scale level.
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=avg_factor)

        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum()

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
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, losses_dfl,\
            avg_factor = multi_apply(
                self.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.prior_generator.strides,
                avg_factor=avg_factor)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. GFL head does not need this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (:obj: `ConfigDict`): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
            is False and mlvl_score_factor is None, return mlvl_bboxes and
            mlvl_scores, else return mlvl_bboxes, mlvl_scores and
            mlvl_score_factor. Usually with_nms is False is used for aug
            test. If with_nms is True, then return the following format

            - det_bboxes (Tensor): Predicted bboxes with shape
              [num_bboxes, 5], where the first 4 columns are bounding
              box positions (tl_x, tl_y, br_x, br_y) and the 5-th
              column are scores between 0 and 1.
            - det_labels (Tensor): Predicted labels of the corresponding
              box with shape [num_bboxes].
        """
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, stride, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list,
                    self.prior_generator.strides, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            bboxes = self.bbox_coder.decode(
                self.anchor_center(priors), bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        results = InstanceData()
        results.bboxes = torch.cat(mlvl_bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    def get_targets(self,
                    anchor_list: List[Tensor],
                    valid_flag_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs=True) -> tuple:
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list,
         sampling_results_list) = multi_apply(
             self._get_targets_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             batch_gt_instances,
             batch_img_metas,
             batch_gt_instances_ignore,
             unmap_outputs=unmap_outputs)
        # Get `avg_factor` of all images, which calculate in `SamplingResult`.
        # When using sampling method, avg_factor is usually the sum of
        # positive and negative priors. When using `PseudoSampler`,
        # `avg_factor` is usually equal to the number of positive priors.
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, avg_factor)

    def _get_targets_single(self,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            num_level_anchors: List[int],
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors (list[int]): Number of anchors of each scale
                level.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.

        Returns:
            tuple: N is the number of total anchors in the image.

            - anchors (Tensor): All anchors in the image with shape (N, 4).
            - labels (Tensor): Labels of all anchors in the image with
              shape (N,).
            - label_weights (Tensor): Label weights of all anchor in the
              image with shape (N,).
            - bbox_targets (Tensor): BBox targets of all anchors in the
              image with shape (N, 4).
            - bbox_weights (Tensor): BBox weights of all anchors in the
              image with shape (N, 4).
            - pos_inds (Tensor): Indices of positive anchor with shape
              (num_pos,).
            - neg_inds (Tensor): Indices of negative anchor with shape
              (num_neg,).
            - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            num_level_priors=num_level_anchors_inside,
            gt_instances=gt_instances,
            gt_instances_ignore=gt_instances_ignore)

        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
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

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, sampling_result)

    def get_num_level_anchors_inside(self, num_level_anchors: List[int],
                                     inside_flags: Tensor) -> List[int]:
        """Get the number of valid anchors in every level."""

        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
