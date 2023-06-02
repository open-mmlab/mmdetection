# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList)
from ..task_modules.samplers import PseudoSampler
from ..utils import (filter_scores_and_topk, images_to_levels, multi_apply,
                     unmap)
from .base_dense_head import BaseDenseHead
from .guided_anchor_head import GuidedAnchorHead


@MODELS.register_module()
class SABLRetinaHead(BaseDenseHead):
    """Side-Aware Boundary Localization (SABL) for RetinaNet.

    The anchor generation, assigning and sampling in SABLRetinaHead
    are the same as GuidedAnchorHead for guided anchoring.

    Please refer to https://arxiv.org/abs/1912.04260 for more details.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of Convs for classification and
            regression branches. Defaults to 4.
        feat_channels (int): Number of hidden channels. Defaults to 256.
        approx_anchor_generator (:obj:`ConfigType` or dict): Config dict for
            approx generator.
        square_anchor_generator (:obj:`ConfigDict` or dict): Config dict for
            square generator.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            ConvModule. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            Norm Layer. Defaults to None.
        bbox_coder (:obj:`ConfigDict` or dict): Config dict for bbox coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be ``True`` when
            using ``IoULoss``, ``GIoULoss``, or ``DIoULoss`` in the bbox head.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            SABLRetinaHead.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            SABLRetinaHead.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox_cls (:obj:`ConfigDict` or dict): Config of classification
            loss for bbox branch.
        loss_bbox_reg (:obj:`ConfigDict` or dict): Config of regression loss
            for bbox branch.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        stacked_convs: int = 4,
        feat_channels: int = 256,
        approx_anchor_generator: ConfigType = dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        square_anchor_generator: ConfigType = dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[4],
            strides=[8, 16, 32, 64, 128]),
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        bbox_coder: ConfigType = dict(
            type='BucketingBBoxCoder', num_buckets=14, scale_factor=3.0),
        reg_decoded_bbox: bool = False,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        loss_cls: ConfigType = dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_cls: ConfigType = dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.5),
        loss_bbox_reg: ConfigType = dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.5),
        init_cfg: MultiConfig = dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal', name='retina_cls', std=0.01, bias_prob=0.01))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_buckets = bbox_coder['num_buckets']
        self.side_num = int(np.ceil(self.num_buckets / 2))

        assert (approx_anchor_generator['octave_base_scale'] ==
                square_anchor_generator['scales'][0])
        assert (approx_anchor_generator['strides'] ==
                square_anchor_generator['strides'])

        self.approx_anchor_generator = TASK_UTILS.build(
            approx_anchor_generator)
        self.square_anchor_generator = TASK_UTILS.build(
            square_anchor_generator)
        self.approxs_per_octave = (
            self.approx_anchor_generator.num_base_priors[0])

        # one anchor per location
        self.num_base_priors = self.square_anchor_generator.num_base_priors[0]

        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.reg_decoded_bbox = reg_decoded_bbox

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox_cls = MODELS.build(loss_bbox_cls)
        self.loss_bbox_reg = MODELS.build(loss_bbox_reg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            # use PseudoSampler when sampling is False
            if 'sampler' in self.train_cfg:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'], default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler(context=self)

        self._init_layers()

    def _init_layers(self) -> None:
        self.relu = nn.ReLU(inplace=True)
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
        self.retina_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.retina_bbox_reg = nn.Conv2d(
            self.feat_channels, self.side_num * 4, 3, padding=1)
        self.retina_bbox_cls = nn.Conv2d(
            self.feat_channels, self.side_num * 4, 3, padding=1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_cls_pred = self.retina_bbox_cls(reg_feat)
        bbox_reg_pred = self.retina_bbox_reg(reg_feat)
        bbox_pred = (bbox_cls_pred, bbox_reg_pred)
        return cls_score, bbox_pred

    def forward(self, feats: List[Tensor]) -> Tuple[List[Tensor]]:
        return multi_apply(self.forward_single, feats)

    def get_anchors(
        self,
        featmap_sizes: List[tuple],
        img_metas: List[dict],
        device: Union[torch.device, str] = 'cuda'
    ) -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
        """Get squares according to feature map sizes and guided anchors.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: square approxs of each image
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # squares for one time
        multi_level_squares = self.square_anchor_generator.grid_priors(
            featmap_sizes, device=device)
        squares_list = [multi_level_squares for _ in range(num_imgs)]

        return squares_list

    def get_targets(self,
                    approx_list: List[List[Tensor]],
                    inside_flag_list: List[List[Tensor]],
                    square_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas,
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs=True) -> tuple:
        """Compute bucketing targets.

        Args:
            approx_list (list[list[Tensor]]): Multi level approxs of each
                image.
            inside_flag_list (list[list[Tensor]]): Multi level inside flags of
                each image.
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
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.

        Returns:
            tuple: Returns a tuple containing learning targets.

            - labels_list (list[Tensor]): Labels of each level.
            - label_weights_list (list[Tensor]): Label weights of each level.
            - bbox_cls_targets_list (list[Tensor]): BBox cls targets of \
            each level.
            - bbox_cls_weights_list (list[Tensor]): BBox cls weights of \
            each level.
            - bbox_reg_targets_list (list[Tensor]): BBox reg targets of \
            each level.
            - bbox_reg_weights_list (list[Tensor]): BBox reg weights of \
            each level.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
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
        (all_labels, all_label_weights, all_bbox_cls_targets,
         all_bbox_cls_weights, all_bbox_reg_targets, all_bbox_reg_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = multi_apply(
             self._get_targets_single,
             approx_flat_list,
             inside_flag_flat_list,
             square_flat_list,
             batch_gt_instances,
             batch_img_metas,
             batch_gt_instances_ignore,
             unmap_outputs=unmap_outputs)

        # sampled anchors of all images
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_squares)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_squares)
        bbox_cls_targets_list = images_to_levels(all_bbox_cls_targets,
                                                 num_level_squares)
        bbox_cls_weights_list = images_to_levels(all_bbox_cls_weights,
                                                 num_level_squares)
        bbox_reg_targets_list = images_to_levels(all_bbox_reg_targets,
                                                 num_level_squares)
        bbox_reg_weights_list = images_to_levels(all_bbox_reg_weights,
                                                 num_level_squares)
        return (labels_list, label_weights_list, bbox_cls_targets_list,
                bbox_cls_weights_list, bbox_reg_targets_list,
                bbox_reg_weights_list, avg_factor)

    def _get_targets_single(self,
                            flat_approxs: Tensor,
                            inside_flags: Tensor,
                            flat_squares: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_approxs (Tensor): flat approxs of a single image,
                shape (n, 4)
            inside_flags (Tensor): inside flags of a single image,
                shape (n, ).
            flat_squares (Tensor): flat squares of a single image,
                shape (approxs_per_octave * n, 4)
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.  Defaults to True.

        Returns:
            tuple:

            - labels_list (Tensor): Labels in a single image.
            - label_weights (Tensor): Label weights in a single image.
            - bbox_cls_targets (Tensor): BBox cls targets in a single image.
            - bbox_cls_weights (Tensor): BBox cls weights in a single image.
            - bbox_reg_targets (Tensor): BBox reg targets in a single image.
            - bbox_reg_weights (Tensor): BBox reg weights in a single image.
            - num_total_pos (int): Number of positive samples in a single \
            image.
            - num_total_neg (int): Number of negative samples in a single \
            image.
            - sampling_result (:obj:`SamplingResult`): Sampling result object.
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
        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_squares = squares.shape[0]
        bbox_cls_targets = squares.new_zeros(
            (num_valid_squares, self.side_num * 4))
        bbox_cls_weights = squares.new_zeros(
            (num_valid_squares, self.side_num * 4))
        bbox_reg_targets = squares.new_zeros(
            (num_valid_squares, self.side_num * 4))
        bbox_reg_weights = squares.new_zeros(
            (num_valid_squares, self.side_num * 4))
        labels = squares.new_full((num_valid_squares, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = squares.new_zeros(num_valid_squares, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            (pos_bbox_reg_targets, pos_bbox_reg_weights, pos_bbox_cls_targets,
             pos_bbox_cls_weights) = self.bbox_coder.encode(
                 sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)

            bbox_cls_targets[pos_inds, :] = pos_bbox_cls_targets
            bbox_reg_targets[pos_inds, :] = pos_bbox_reg_targets
            bbox_cls_weights[pos_inds, :] = pos_bbox_cls_weights
            bbox_reg_weights[pos_inds, :] = pos_bbox_reg_weights
            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_squares.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_cls_targets = unmap(bbox_cls_targets, num_total_anchors,
                                     inside_flags)
            bbox_cls_weights = unmap(bbox_cls_weights, num_total_anchors,
                                     inside_flags)
            bbox_reg_targets = unmap(bbox_reg_targets, num_total_anchors,
                                     inside_flags)
            bbox_reg_weights = unmap(bbox_reg_weights, num_total_anchors,
                                     inside_flags)
        return (labels, label_weights, bbox_cls_targets, bbox_cls_weights,
                bbox_reg_targets, bbox_reg_weights, pos_inds, neg_inds,
                sampling_result)

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            labels: Tensor, label_weights: Tensor,
                            bbox_cls_targets: Tensor, bbox_cls_weights: Tensor,
                            bbox_reg_targets: Tensor, bbox_reg_weights: Tensor,
                            avg_factor: float) -> Tuple[Tensor]:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels in a single image.
            label_weights (Tensor): Label weights in a single level.
            bbox_cls_targets (Tensor): BBox cls targets in a single level.
            bbox_cls_weights (Tensor): BBox cls weights in a single level.
            bbox_reg_targets (Tensor): BBox reg targets in a single level.
            bbox_reg_weights (Tensor): BBox reg weights in a single level.
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)
        # regression loss
        bbox_cls_targets = bbox_cls_targets.reshape(-1, self.side_num * 4)
        bbox_cls_weights = bbox_cls_weights.reshape(-1, self.side_num * 4)
        bbox_reg_targets = bbox_reg_targets.reshape(-1, self.side_num * 4)
        bbox_reg_weights = bbox_reg_weights.reshape(-1, self.side_num * 4)
        (bbox_cls_pred, bbox_reg_pred) = bbox_pred
        bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(
            -1, self.side_num * 4)
        bbox_reg_pred = bbox_reg_pred.permute(0, 2, 3, 1).reshape(
            -1, self.side_num * 4)
        loss_bbox_cls = self.loss_bbox_cls(
            bbox_cls_pred,
            bbox_cls_targets.long(),
            bbox_cls_weights,
            avg_factor=avg_factor * 4 * self.side_num)
        loss_bbox_reg = self.loss_bbox_reg(
            bbox_reg_pred,
            bbox_reg_targets,
            bbox_reg_weights,
            avg_factor=avg_factor * 4 * self.bbox_coder.offset_topk)
        return loss_cls, loss_bbox_cls, loss_bbox_reg

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
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
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

        # get sampled approxes
        approxs_list, inside_flag_list = GuidedAnchorHead.get_sampled_approxs(
            self, featmap_sizes, batch_img_metas, device=device)

        square_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            approxs_list,
            inside_flag_list,
            square_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_cls_targets_list,
         bbox_cls_weights_list, bbox_reg_targets_list, bbox_reg_weights_list,
         avg_factor) = cls_reg_targets

        losses_cls, losses_bbox_cls, losses_bbox_reg = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_cls_targets_list,
            bbox_cls_weights_list,
            bbox_reg_targets_list,
            bbox_reg_weights_list,
            avg_factor=avg_factor)
        return dict(
            loss_cls=losses_cls,
            loss_bbox_cls=losses_bbox_cls,
            loss_bbox_reg=losses_bbox_reg)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        batch_img_metas: List[dict],
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
            cfg (:obj:`ConfigDict`, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

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
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        device = cls_scores[0].device
        mlvl_anchors = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_cls_pred_list = [
                bbox_preds[i][0][img_id].detach() for i in range(num_levels)
            ]
            bbox_reg_pred_list = [
                bbox_preds[i][1][img_id].detach() for i in range(num_levels)
            ]
            proposals = self._predict_by_feat_single(
                cls_scores=cls_score_list,
                bbox_cls_preds=bbox_cls_pred_list,
                bbox_reg_preds=bbox_reg_pred_list,
                mlvl_anchors=mlvl_anchors[img_id],
                img_meta=batch_img_metas[img_id],
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(proposals)
        return result_list

    def _predict_by_feat_single(self,
                                cls_scores: List[Tensor],
                                bbox_cls_preds: List[Tensor],
                                bbox_reg_preds: List[Tensor],
                                mlvl_anchors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        cfg = self.test_cfg if cfg is None else cfg
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_confids = []
        mlvl_labels = []
        assert len(cls_scores) == len(bbox_cls_preds) == len(
            bbox_reg_preds) == len(mlvl_anchors)
        for cls_score, bbox_cls_pred, bbox_reg_pred, anchors in zip(
                cls_scores, bbox_cls_preds, bbox_reg_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_cls_pred.size(
            )[-2:] == bbox_reg_pred.size()[-2::]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]
            bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(
                -1, self.side_num * 4)
            bbox_reg_pred = bbox_reg_pred.permute(1, 2, 0).reshape(
                -1, self.side_num * 4)

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(
                    anchors=anchors,
                    bbox_cls_pred=bbox_cls_pred,
                    bbox_reg_pred=bbox_reg_pred))
            scores, labels, _, filtered_results = results

            anchors = filtered_results['anchors']
            bbox_cls_pred = filtered_results['bbox_cls_pred']
            bbox_reg_pred = filtered_results['bbox_reg_pred']

            bbox_preds = [
                bbox_cls_pred.contiguous(),
                bbox_reg_pred.contiguous()
            ]
            bboxes, confids = self.bbox_coder.decode(
                anchors.contiguous(),
                bbox_preds,
                max_shape=img_meta['img_shape'])

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_confids.append(confids)
            mlvl_labels.append(labels)

        results = InstanceData()
        results.bboxes = torch.cat(mlvl_bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.score_factors = torch.cat(mlvl_confids)
        results.labels = torch.cat(mlvl_labels)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
