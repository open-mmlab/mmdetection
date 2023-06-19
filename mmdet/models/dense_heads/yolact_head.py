# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig)
from ..layers import fast_nms
from ..utils import images_to_levels, multi_apply, select_single_mlvl
from ..utils.misc import empty_instances
from .anchor_head import AnchorHead
from .base_mask_head import BaseMaskHead


@MODELS.register_module()
class YOLACTHead(AnchorHead):
    """YOLACT box head used in https://arxiv.org/abs/1904.02689.

    Note that YOLACT head is a light version of RetinaNet head.
    Four differences are described as follows:

    1. YOLACT box head has three-times fewer anchors.
    2. YOLACT box head shares the convs for box and cls branches.
    3. YOLACT box head uses OHEM instead of Focal loss.
    4. YOLACT box head predicts a set of mask coefficients for each box.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        anchor_generator (:obj:`ConfigDict` or dict): Config dict for
            anchor generator
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        num_head_convs (int): Number of the conv layers shared by
            box and cls branches.
        num_protos (int): Number of the mask coefficients.
        use_ohem (bool): If true, ``loss_single_OHEM`` will be used for
            cls loss calculation. If false, ``loss_single`` will be used.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Dictionary to
            construct and config conv layer.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Dictionary to
            construct and config norm layer.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 anchor_generator: ConfigType = dict(
                     type='AnchorGenerator',
                     octave_base_scale=3,
                     scales_per_octave=1,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     reduction='none',
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.5),
                 num_head_convs: int = 1,
                 num_protos: int = 32,
                 use_ohem: bool = True,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = dict(
                     type='Xavier',
                     distribution='uniform',
                     bias=0,
                     layer='Conv2d'),
                 **kwargs) -> None:
        self.num_head_convs = num_head_convs
        self.num_protos = num_protos
        self.use_ohem = use_ohem
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.head_convs = ModuleList()
        for i in range(self.num_head_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.head_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.conv_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.conv_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)
        self.conv_coeff = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.num_protos,
            3,
            padding=1)

    def forward_single(self, x: Tensor) -> tuple:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:

            - cls_score (Tensor): Cls scores for a single scale level
              the channels number is num_anchors * num_classes.
            - bbox_pred (Tensor): Box energies / deltas for a single scale
              level, the channels number is num_anchors * 4.
            - coeff_pred (Tensor): Mask coefficients for a single scale
              level, the channels number is num_anchors * num_protos.
        """
        for head_conv in self.head_convs:
            x = head_conv(x)
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        coeff_pred = self.conv_coeff(x).tanh()
        return cls_score, bbox_pred, coeff_pred

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            coeff_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        When ``self.use_ohem == True``, it functions like ``SSDHead.loss``,
        otherwise, it follows ``AnchorHead.loss``.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            coeff_preds (list[Tensor]): Mask coefficients for each scale
                level with shape (N, num_anchors * num_protos, H, W)
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
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            unmap_outputs=not self.use_ohem,
            return_sampling_results=True)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor, sampling_results) = cls_reg_targets

        if self.use_ohem:
            num_images = len(batch_img_metas)
            all_cls_scores = torch.cat([
                s.permute(0, 2, 3, 1).reshape(
                    num_images, -1, self.cls_out_channels) for s in cls_scores
            ], 1)
            all_labels = torch.cat(labels_list, -1).view(num_images, -1)
            all_label_weights = torch.cat(label_weights_list,
                                          -1).view(num_images, -1)
            all_bbox_preds = torch.cat([
                b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
                for b in bbox_preds
            ], -2)
            all_bbox_targets = torch.cat(bbox_targets_list,
                                         -2).view(num_images, -1, 4)
            all_bbox_weights = torch.cat(bbox_weights_list,
                                         -2).view(num_images, -1, 4)

            # concat all level anchors to a single tensor
            all_anchors = []
            for i in range(num_images):
                all_anchors.append(torch.cat(anchor_list[i]))

            # check NaN and Inf
            assert torch.isfinite(all_cls_scores).all().item(), \
                'classification scores become infinite or NaN!'
            assert torch.isfinite(all_bbox_preds).all().item(), \
                'bbox predications become infinite or NaN!'

            losses_cls, losses_bbox = multi_apply(
                self.OHEMloss_by_feat_single,
                all_cls_scores,
                all_bbox_preds,
                all_anchors,
                all_labels,
                all_label_weights,
                all_bbox_targets,
                all_bbox_weights,
                avg_factor=avg_factor)
        else:
            # anchor number of multi levels
            num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
            # concat all level anchors and flags to a single tensor
            concat_anchor_list = []
            for i in range(len(anchor_list)):
                concat_anchor_list.append(torch.cat(anchor_list[i]))
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
        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        # update `_raw_positive_infos`, which will be used when calling
        # `get_positive_infos`.
        self._raw_positive_infos.update(coeff_preds=coeff_preds)
        return losses

    def OHEMloss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                                anchors: Tensor, labels: Tensor,
                                label_weights: Tensor, bbox_targets: Tensor,
                                bbox_weights: Tensor,
                                avg_factor: int) -> tuple:
        """Compute loss of a single image. Similar to
        func:``SSDHead.loss_by_feat_single``

        Args:
            cls_score (Tensor): Box scores for eachimage
                Has shape (num_total_anchors, num_classes).
            bbox_pred (Tensor): Box energies / deltas for each image
                level with shape (num_total_anchors, 4).
            anchors (Tensor): Box reference for each scale level with shape
                (num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape
                (num_total_anchors,)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            Tuple[Tensor, Tensor]: A tuple of cls loss and bbox loss of one
            feature map.
        """

        loss_cls_all = self.loss_cls(cls_score, labels, label_weights)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(
            as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(
            as_tuple=False).view(-1)

        num_pos_samples = pos_inds.size(0)
        if num_pos_samples == 0:
            num_neg_samples = neg_inds.size(0)
        else:
            num_neg_samples = self.train_cfg['neg_pos_ratio'] * \
                              num_pos_samples
            if num_neg_samples > neg_inds.size(0):
                num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / avg_factor
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        return loss_cls[None], loss_bbox

    def get_positive_infos(self) -> InstanceList:
        """Get positive information from sampling results.

        Returns:
            list[:obj:`InstanceData`]: Positive Information of each image,
            usually including positive bboxes, positive labels, positive
            priors, positive coeffs, etc.
        """
        assert len(self._raw_positive_infos) > 0
        sampling_results = self._raw_positive_infos['sampling_results']
        num_imgs = len(sampling_results)

        coeff_pred_list = []
        for coeff_pred_per_level in self._raw_positive_infos['coeff_preds']:
            coeff_pred_per_level = \
                coeff_pred_per_level.permute(
                    0, 2, 3, 1).reshape(num_imgs, -1, self.num_protos)
            coeff_pred_list.append(coeff_pred_per_level)
        coeff_preds = torch.cat(coeff_pred_list, dim=1)

        pos_info_list = []
        for idx, sampling_result in enumerate(sampling_results):
            pos_info = InstanceData()
            coeff_preds_single = coeff_preds[idx]
            pos_info.pos_assigned_gt_inds = \
                sampling_result.pos_assigned_gt_inds
            pos_info.pos_inds = sampling_result.pos_inds
            pos_info.coeffs = coeff_preds_single[sampling_result.pos_inds]
            pos_info.bboxes = sampling_result.pos_gt_bboxes
            pos_info_list.append(pos_info)
        return pos_info_list

    def predict_by_feat(self,
                        cls_scores,
                        bbox_preds,
                        coeff_preds,
                        batch_img_metas,
                        cfg=None,
                        rescale=True,
                        **kwargs):
        """Similar to func:``AnchorHead.get_bboxes``, but additionally
        processes coeff_preds.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            coeff_preds (list[Tensor]): Mask coefficients for each scale
                level with shape (N, num_anchors * num_protos, H, W)
            batch_img_metas (list[dict]): Batch image meta info.
            cfg (:obj:`Config` | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
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
                - coeffs (Tensor): the predicted mask coefficients of
                  instance inside the corresponding box has a shape
                  (n, num_protos).
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            coeff_pred_list = select_single_mlvl(coeff_preds, img_id)
            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                coeff_preds_list=coeff_pred_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                coeff_preds_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigType,
                                rescale: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results. Similar to func:``AnchorHead._predict_by_feat_single``,
        but additionally processes coeff_preds_list and uses fast NMS instead
        of traditional NMS.

        Args:
            cls_score_list (list[Tensor]): Box scores for a single scale level
                Has shape (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_priors * 4, H, W).
            coeff_preds_list (list[Tensor]): Mask coefficients for a single
                scale level with shape (num_priors * num_protos, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid,
                has shape (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
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
                - coeffs (Tensor): the predicted mask coefficients of
                  instance inside the corresponding box has a shape
                  (n, num_protos).
        """
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_priors)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_coeffs = []
        for cls_score, bbox_pred, coeff_pred, priors in \
                zip(cls_score_list, bbox_pred_list,
                    coeff_preds_list, mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            coeff_pred = coeff_pred.permute(1, 2,
                                            0).reshape(-1, self.num_protos)

            if 0 < nms_pre < scores.shape[0]:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                priors = priors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                coeff_pred = coeff_pred[topk_inds, :]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_coeffs.append(coeff_pred)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = torch.cat(mlvl_valid_priors)
        multi_bboxes = self.bbox_coder.decode(
            priors, bbox_pred, max_shape=img_shape)

        multi_scores = torch.cat(mlvl_scores)
        multi_coeffs = torch.cat(mlvl_coeffs)

        return self._bbox_post_process(
            multi_bboxes=multi_bboxes,
            multi_scores=multi_scores,
            multi_coeffs=multi_coeffs,
            cfg=cfg,
            rescale=rescale,
            img_meta=img_meta)

    def _bbox_post_process(self,
                           multi_bboxes: Tensor,
                           multi_scores: Tensor,
                           multi_coeffs: Tensor,
                           cfg: ConfigType,
                           rescale: bool = False,
                           img_meta: Optional[dict] = None,
                           **kwargs) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            multi_bboxes (Tensor): Predicted bbox that concat all levels.
            multi_scores (Tensor): Bbox scores that concat all levels.
            multi_coeffs (Tensor): Mask coefficients  that concat all levels.
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            img_meta (dict, optional): Image meta info. Defaults to None.

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
                - coeffs (Tensor): the predicted mask coefficients of
                  instance inside the corresponding box has a shape
                  (n, num_protos).
        """
        if rescale:
            assert img_meta.get('scale_factor') is not None
            multi_bboxes /= multi_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))
            # mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class

            padding = multi_scores.new_zeros(multi_scores.shape[0], 1)
            multi_scores = torch.cat([multi_scores, padding], dim=1)
        det_bboxes, det_labels, det_coeffs = fast_nms(
            multi_bboxes, multi_scores, multi_coeffs, cfg.score_thr,
            cfg.iou_thr, cfg.top_k, cfg.max_per_img)
        results = InstanceData()
        results.bboxes = det_bboxes[:, :4]
        results.scores = det_bboxes[:, -1]
        results.labels = det_labels
        results.coeffs = det_coeffs
        return results


@MODELS.register_module()
class YOLACTProtonet(BaseMaskHead):
    """YOLACT mask head used in https://arxiv.org/abs/1904.02689.

    This head outputs the mask prototypes for YOLACT.

    Args:
        in_channels (int): Number of channels in the input feature map.
        proto_channels (tuple[int]): Output channels of protonet convs.
        proto_kernel_sizes (tuple[int]): Kernel sizes of protonet convs.
        include_last_relu (bool): If keep the last relu of protonet.
        num_protos (int): Number of prototypes.
        num_classes (int): Number of categories excluding the background
            category.
        loss_mask_weight (float): Reweight the mask loss by this factor.
        max_masks_to_train (int): Maximum number of masks to train for
            each image.
        with_seg_branch (bool): Whether to apply a semantic segmentation
            branch and calculate loss during training to increase
            performance with no speed penalty. Defaults to True.
        loss_segm (:obj:`ConfigDict` or dict, optional): Config of
            semantic segmentation loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config
            of head.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            head.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        proto_channels: tuple = (256, 256, 256, None, 256, 32),
        proto_kernel_sizes: tuple = (3, 3, 3, -2, 3, 1),
        include_last_relu: bool = True,
        num_protos: int = 32,
        loss_mask_weight: float = 1.0,
        max_masks_to_train: int = 100,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        with_seg_branch: bool = True,
        loss_segm: ConfigType = dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        init_cfg=dict(
            type='Xavier',
            distribution='uniform',
            override=dict(name='protonet'))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.proto_channels = proto_channels
        self.proto_kernel_sizes = proto_kernel_sizes
        self.include_last_relu = include_last_relu

        # Segmentation branch
        self.with_seg_branch = with_seg_branch
        self.segm_branch = SegmentationModule(
            num_classes=num_classes, in_channels=in_channels) \
            if with_seg_branch else None
        self.loss_segm = MODELS.build(loss_segm) if with_seg_branch else None

        self.loss_mask_weight = loss_mask_weight
        self.num_protos = num_protos
        self.num_classes = num_classes
        self.max_masks_to_train = max_masks_to_train
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        # Possible patterns:
        # ( 256, 3) -> conv
        # ( 256,-2) -> deconv
        # (None,-2) -> bilinear interpolate
        in_channels = self.in_channels
        protonets = ModuleList()
        for num_channels, kernel_size in zip(self.proto_channels,
                                             self.proto_kernel_sizes):
            if kernel_size > 0:
                layer = nn.Conv2d(
                    in_channels,
                    num_channels,
                    kernel_size,
                    padding=kernel_size // 2)
            else:
                if num_channels is None:
                    layer = InterpolateModule(
                        scale_factor=-kernel_size,
                        mode='bilinear',
                        align_corners=False)
                else:
                    layer = nn.ConvTranspose2d(
                        in_channels,
                        num_channels,
                        -kernel_size,
                        padding=kernel_size // 2)
            protonets.append(layer)
            protonets.append(nn.ReLU(inplace=True))
            in_channels = num_channels if num_channels is not None \
                else in_channels
        if not self.include_last_relu:
            protonets = protonets[:-1]
        self.protonet = nn.Sequential(*protonets)

    def forward(self, x: tuple, positive_infos: InstanceList) -> tuple:
        """Forward feature from the upstream network to get prototypes and
        linearly combine the prototypes, using masks coefficients, into
        instance masks. Finally, crop the instance masks with given bboxes.

        Args:
            x (Tuple[Tensor]): Feature from the upstream network, which is
                a 4D-tensor.
            positive_infos (List[:obj:``InstanceData``]): Positive information
                that calculate from detect head.

        Returns:
            tuple: Predicted instance segmentation masks and
            semantic segmentation map.
        """
        # YOLACT used single feature map to get segmentation masks
        single_x = x[0]

        # YOLACT segmentation branch, if not training or segmentation branch
        # is None, will not process the forward function.
        if self.segm_branch is not None and self.training:
            segm_preds = self.segm_branch(single_x)
        else:
            segm_preds = None
        # YOLACT mask head
        prototypes = self.protonet(single_x)
        prototypes = prototypes.permute(0, 2, 3, 1).contiguous()

        num_imgs = single_x.size(0)

        mask_pred_list = []
        for idx in range(num_imgs):
            cur_prototypes = prototypes[idx]
            pos_coeffs = positive_infos[idx].coeffs

            # Linearly combine the prototypes with the mask coefficients
            mask_preds = cur_prototypes @ pos_coeffs.t()
            mask_preds = torch.sigmoid(mask_preds)
            mask_pred_list.append(mask_preds)
        return mask_pred_list, segm_preds

    def loss_by_feat(self, mask_preds: List[Tensor], segm_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict], positive_infos: InstanceList,
                     **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the mask head.

        Args:
            mask_preds (list[Tensor]): List of predicted prototypes, each has
                shape (num_classes, H, W).
            segm_preds (Tensor):  Predicted semantic segmentation map with
                shape (N, num_classes, H, W)
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``masks``,
                and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of multiple images.
            positive_infos (List[:obj:``InstanceData``]): Information of
                positive samples of each image that are assigned in detection
                head.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert positive_infos is not None, \
            'positive_infos should not be None in `YOLACTProtonet`'
        losses = dict()

        # crop
        croped_mask_pred = self.crop_mask_preds(mask_preds, batch_img_metas,
                                                positive_infos)

        loss_mask = []
        loss_segm = []
        num_imgs, _, mask_h, mask_w = segm_preds.size()
        assert num_imgs == len(croped_mask_pred)
        segm_avg_factor = num_imgs * mask_h * mask_w
        total_pos = 0

        if self.segm_branch is not None:
            assert segm_preds is not None

        for idx in range(num_imgs):
            img_meta = batch_img_metas[idx]

            (mask_preds, pos_mask_targets, segm_targets, num_pos,
             gt_bboxes_for_reweight) = self._get_targets_single(
                 croped_mask_pred[idx], segm_preds[idx],
                 batch_gt_instances[idx], positive_infos[idx])

            # segmentation loss
            if self.with_seg_branch:
                if segm_targets is None:
                    loss = segm_preds[idx].sum() * 0.
                else:
                    loss = self.loss_segm(
                        segm_preds[idx],
                        segm_targets,
                        avg_factor=segm_avg_factor)
                loss_segm.append(loss)
            # mask loss
            total_pos += num_pos
            if num_pos == 0 or pos_mask_targets is None:
                loss = mask_preds.sum() * 0.
            else:
                mask_preds = torch.clamp(mask_preds, 0, 1)
                loss = F.binary_cross_entropy(
                    mask_preds, pos_mask_targets,
                    reduction='none') * self.loss_mask_weight

                h, w = img_meta['img_shape'][:2]
                gt_bboxes_width = (gt_bboxes_for_reweight[:, 2] -
                                   gt_bboxes_for_reweight[:, 0]) / w
                gt_bboxes_height = (gt_bboxes_for_reweight[:, 3] -
                                    gt_bboxes_for_reweight[:, 1]) / h
                loss = loss.mean(dim=(1,
                                      2)) / gt_bboxes_width / gt_bboxes_height
                loss = torch.sum(loss)
            loss_mask.append(loss)

        if total_pos == 0:
            total_pos += 1  # avoid nan
        loss_mask = [x / total_pos for x in loss_mask]

        losses.update(loss_mask=loss_mask)
        if self.with_seg_branch:
            losses.update(loss_segm=loss_segm)

        return losses

    def _get_targets_single(self, mask_preds: Tensor, segm_pred: Tensor,
                            gt_instances: InstanceData,
                            positive_info: InstanceData):
        """Compute targets for predictions of single image.

        Args:
            mask_preds (Tensor): Predicted prototypes with shape
                (num_classes, H, W).
            segm_pred (Tensor): Predicted semantic segmentation map
                with shape (num_classes, H, W).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes``, ``labels``,
                and ``masks`` attributes.
            positive_info (:obj:`InstanceData`): Information of positive
                samples that are assigned in detection head. It usually
                contains following keys.

                    - pos_assigned_gt_inds (Tensor): Assigner GT indexes of
                      positive proposals, has shape (num_pos, )
                    - pos_inds (Tensor): Positive index of image, has
                      shape (num_pos, ).
                    - coeffs (Tensor): Positive mask coefficients
                      with shape (num_pos, num_protos).
                    - bboxes (Tensor): Positive bboxes with shape
                      (num_pos, 4)

        Returns:
            tuple: Usually returns a tuple containing learning targets.

            - mask_preds (Tensor): Positive predicted mask with shape
              (num_pos, mask_h, mask_w).
            - pos_mask_targets (Tensor): Positive mask targets with shape
              (num_pos, mask_h, mask_w).
            - segm_targets (Tensor): Semantic segmentation targets with shape
              (num_classes, segm_h, segm_w).
            - num_pos (int): Positive numbers.
            - gt_bboxes_for_reweight (Tensor): GT bboxes that match to the
              positive priors has shape (num_pos, 4).
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        device = gt_bboxes.device
        gt_masks = gt_instances.masks.to_tensor(
            dtype=torch.bool, device=device).float()
        if gt_masks.size(0) == 0:
            return mask_preds, None, None, 0, None

        # process with semantic segmentation targets
        if segm_pred is not None:
            num_classes, segm_h, segm_w = segm_pred.size()
            with torch.no_grad():
                downsampled_masks = F.interpolate(
                    gt_masks.unsqueeze(0), (segm_h, segm_w),
                    mode='bilinear',
                    align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()
                segm_targets = torch.zeros_like(segm_pred, requires_grad=False)
                for obj_idx in range(downsampled_masks.size(0)):
                    segm_targets[gt_labels[obj_idx] - 1] = torch.max(
                        segm_targets[gt_labels[obj_idx] - 1],
                        downsampled_masks[obj_idx])
        else:
            segm_targets = None
        # process with mask targets
        pos_assigned_gt_inds = positive_info.pos_assigned_gt_inds
        num_pos = pos_assigned_gt_inds.size(0)
        # Since we're producing (near) full image masks,
        # it'd take too much vram to backprop on every single mask.
        # Thus we select only a subset.
        if num_pos > self.max_masks_to_train:
            perm = torch.randperm(num_pos)
            select = perm[:self.max_masks_to_train]
            mask_preds = mask_preds[select]
            pos_assigned_gt_inds = pos_assigned_gt_inds[select]
            num_pos = self.max_masks_to_train

        gt_bboxes_for_reweight = gt_bboxes[pos_assigned_gt_inds]

        mask_h, mask_w = mask_preds.shape[-2:]
        gt_masks = F.interpolate(
            gt_masks.unsqueeze(0), (mask_h, mask_w),
            mode='bilinear',
            align_corners=False).squeeze(0)
        gt_masks = gt_masks.gt(0.5).float()
        pos_mask_targets = gt_masks[pos_assigned_gt_inds]

        return (mask_preds, pos_mask_targets, segm_targets, num_pos,
                gt_bboxes_for_reweight)

    def crop_mask_preds(self, mask_preds: List[Tensor],
                        batch_img_metas: List[dict],
                        positive_infos: InstanceList) -> list:
        """Crop predicted masks by zeroing out everything not in the predicted
        bbox.

        Args:
            mask_preds (list[Tensor]): Predicted prototypes with shape
                (num_classes, H, W).
            batch_img_metas (list[dict]): Meta information of multiple images.
            positive_infos (List[:obj:``InstanceData``]): Positive
                information that calculate from detect head.

        Returns:
            list: The cropped masks.
        """
        croped_mask_preds = []
        for img_meta, mask_preds, cur_info in zip(batch_img_metas, mask_preds,
                                                  positive_infos):
            bboxes_for_cropping = copy.deepcopy(cur_info.bboxes)
            h, w = img_meta['img_shape'][:2]
            bboxes_for_cropping[:, 0::2] /= w
            bboxes_for_cropping[:, 1::2] /= h
            mask_preds = self.crop_single(mask_preds, bboxes_for_cropping)
            mask_preds = mask_preds.permute(2, 0, 1).contiguous()
            croped_mask_preds.append(mask_preds)
        return croped_mask_preds

    def crop_single(self,
                    masks: Tensor,
                    boxes: Tensor,
                    padding: int = 1) -> Tensor:
        """Crop single predicted masks by zeroing out everything not in the
        predicted bbox.

        Args:
            masks (Tensor): Predicted prototypes, has shape [H, W, N].
            boxes (Tensor): Bbox coords in relative point form with
                shape [N, 4].
            padding (int): Image padding size.

        Return:
            Tensor: The cropped masks.
        """
        h, w, n = masks.size()
        x1, x2 = self.sanitize_coordinates(
            boxes[:, 0], boxes[:, 2], w, padding, cast=False)
        y1, y2 = self.sanitize_coordinates(
            boxes[:, 1], boxes[:, 3], h, padding, cast=False)

        rows = torch.arange(
            w, device=masks.device, dtype=x1.dtype).view(1, -1,
                                                         1).expand(h, w, n)
        cols = torch.arange(
            h, device=masks.device, dtype=x1.dtype).view(-1, 1,
                                                         1).expand(h, w, n)

        masks_left = rows >= x1.view(1, 1, -1)
        masks_right = rows < x2.view(1, 1, -1)
        masks_up = cols >= y1.view(1, 1, -1)
        masks_down = cols < y2.view(1, 1, -1)

        crop_mask = masks_left * masks_right * masks_up * masks_down

        return masks * crop_mask.float()

    def sanitize_coordinates(self,
                             x1: Tensor,
                             x2: Tensor,
                             img_size: int,
                             padding: int = 0,
                             cast: bool = True) -> tuple:
        """Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0,
        and x2 <= image_size. Also converts from relative to absolute
        coordinates and casts the results to long tensors.

        Warning: this does things in-place behind the scenes so
        copy if necessary.

        Args:
            x1 (Tensor): shape (N, ).
            x2 (Tensor): shape (N, ).
            img_size (int): Size of the input image.
            padding (int): x1 >= padding, x2 <= image_size-padding.
            cast (bool): If cast is false, the result won't be cast to longs.

        Returns:
            tuple:

            - x1 (Tensor): Sanitized _x1.
            - x2 (Tensor): Sanitized _x2.
        """
        x1 = x1 * img_size
        x2 = x2 * img_size
        if cast:
            x1 = x1.long()
            x2 = x2.long()
        x1 = torch.min(x1, x2)
        x2 = torch.max(x1, x2)
        x1 = torch.clamp(x1 - padding, min=0)
        x2 = torch.clamp(x2 + padding, max=img_size)
        return x1, x2

    def predict_by_feat(self,
                        mask_preds: List[Tensor],
                        segm_preds: Tensor,
                        results_list: InstanceList,
                        batch_img_metas: List[dict],
                        rescale: bool = True,
                        **kwargs) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        mask results.

        Args:
            mask_preds (list[Tensor]): Predicted prototypes with shape
                (num_classes, H, W).
            results_list (List[:obj:``InstanceData``]): BBoxHead results.
            batch_img_metas (list[dict]): Meta information of all images.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

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
        assert len(mask_preds) == len(results_list) == len(batch_img_metas)

        croped_mask_pred = self.crop_mask_preds(mask_preds, batch_img_metas,
                                                results_list)

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = results_list[img_id]
            bboxes = results.bboxes
            mask_preds = croped_mask_pred[img_id]
            if bboxes.shape[0] == 0 or mask_preds.shape[0] == 0:
                results_list[img_id] = empty_instances(
                    [img_meta],
                    bboxes.device,
                    task_type='mask',
                    instance_results=[results])[0]
            else:
                im_mask = self._predict_by_feat_single(
                    mask_preds=croped_mask_pred[img_id],
                    bboxes=bboxes,
                    img_meta=img_meta,
                    rescale=rescale)
                results.masks = im_mask
        return results_list

    def _predict_by_feat_single(self,
                                mask_preds: Tensor,
                                bboxes: Tensor,
                                img_meta: dict,
                                rescale: bool,
                                cfg: OptConfigType = None):
        """Transform a single image's features extracted from the head into
        mask results.

        Args:
            mask_preds (Tensor): Predicted prototypes, has shape [H, W, N].
            bboxes (Tensor): Bbox coords in relative point form with
                shape [N, 4].
            img_meta (dict): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If rescale is False, then returned masks will
                fit the scale of imgs[0].
            cfg (dict, optional): Config used in test phase.
                Defaults to None.

        Returns:
            :obj:`InstanceData`: Processed results of single image.
             it usually contains following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """
        cfg = self.test_cfg if cfg is None else cfg
        scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            (1, 2))
        img_h, img_w = img_meta['ori_shape'][:2]
        if rescale:  # in-placed rescale the bboxes
            scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
                (1, 2))
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)

        masks = F.interpolate(
            mask_preds.unsqueeze(0), (img_h, img_w),
            mode='bilinear',
            align_corners=False).squeeze(0) > cfg.mask_thr

        if cfg.mask_thr_binary < 0:
            # for visualization and debugging
            masks = (masks * 255).to(dtype=torch.uint8)

        return masks


class SegmentationModule(BaseModule):
    """YOLACT segmentation branch used in <https://arxiv.org/abs/1904.02689>`_

    In mmdet v2.x `segm_loss` is calculated in YOLACTSegmHead, while in
    mmdet v3.x `SegmentationModule` is used to obtain the predicted semantic
    segmentation map and `segm_loss` is calculated in YOLACTProtonet.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        init_cfg: ConfigType = dict(
            type='Xavier',
            distribution='uniform',
            override=dict(name='segm_conv'))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.segm_conv = nn.Conv2d(
            self.in_channels, self.num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward feature from the upstream network.

        Args:
            x (Tensor): Feature from the upstream network, which is
                a 4D-tensor.

        Returns:
            Tensor: Predicted semantic segmentation map with shape
                (N, num_classes, H, W).
        """
        return self.segm_conv(x)


class InterpolateModule(BaseModule):
    """This is a module version of F.interpolate.

    Any arguments you give it just get passed along for the ride.
    """

    def __init__(self, *args, init_cfg=None, **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Tensor) -> Tensor:
        """Forward features from the upstream network.

        Args:
            x (Tensor): Feature from the upstream network, which is
                a 4D-tensor.

        Returns:
            Tensor: A 4D-tensor feature map.
        """
        return F.interpolate(x, *self.args, **self.kwargs)
