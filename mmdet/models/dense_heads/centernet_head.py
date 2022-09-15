# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig)
from ..utils import (gaussian_radius, gen_gaussian_target, get_local_maximum,
                     get_topk_from_heatmap, multi_apply,
                     transpose_and_gather_feat)
from .base_dense_head import BaseDenseHead


@MODELS.register_module()
class CenterNetHead(BaseDenseHead):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channels (int): Number of channel in the input feature map.
        feat_channels (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (:obj:`ConfigDict` or dict): Config of center
            heatmap loss. Defaults to
            dict(type='GaussianFocalLoss', loss_weight=1.0)
        loss_wh (:obj:`ConfigDict` or dict): Config of wh loss. Defaults to
             dict(type='L1Loss', loss_weight=0.1).
        loss_offset (:obj:`ConfigDict` or dict): Config of offset loss.
            Defaults to dict(type='L1Loss', loss_weight=1.0).
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config.
            Useless in CenterNet, but we keep this variable for
            SingleStageDetector.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config
            of CenterNet.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization
            config dict.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 num_classes: int,
                 loss_center_heatmap: ConfigType = dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh: ConfigType = dict(type='L1Loss', loss_weight=0.1),
                 loss_offset: ConfigType = dict(
                     type='L1Loss', loss_weight=1.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channels, feat_channels,
                                             num_classes)
        self.wh_head = self._build_head(in_channels, feat_channels, 2)
        self.offset_head = self._build_head(in_channels, feat_channels, 2)

        self.loss_center_heatmap = MODELS.build(loss_center_heatmap)
        self.loss_wh = MODELS.build(loss_wh)
        self.loss_offset = MODELS.build(loss_offset)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_head(self, in_channels: int, feat_channels: int,
                    out_channels: int) -> nn.Sequential:
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, out_channels, kernel_size=1))
        return layer

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[List[Tensor]]:
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (list[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (list[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, x)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward feature of a single level.

        Args:
            x (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(x).sigmoid()
        wh_pred = self.wh_head(x)
        offset_pred = self.offset_head(x)
        return center_heatmap_pred, wh_pred, offset_pred

    def loss_by_feat(
            self,
            center_heatmap_preds: List[Tensor],
            wh_preds: List[Tensor],
            offset_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
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
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]

        gt_bboxes = [
            gt_instances.bboxes for gt_instances in batch_gt_instances
        ]
        gt_labels = [
            gt_instances.labels for gt_instances in batch_gt_instances
        ]
        img_shape = batch_img_metas[0]['batch_input_shape']
        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
                                                     center_heatmap_pred.shape,
                                                     img_shape)

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        # Since the channel of wh_target and offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
        loss_wh = self.loss_wh(
            wh_pred,
            wh_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset)

    def get_targets(self, gt_bboxes: List[Tensor], gt_labels: List[Tensor],
                    feat_shape: tuple, img_shape: tuple) -> Tuple[dict, int]:
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (tuple): feature map shape with value [B, _, H, W]
            img_shape (tuple): image shape.

        Returns:
            tuple[dict, float]: The float value is mean avg_factor, the dict
            has components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight)
        return target_result, avg_factor

    def predict_by_feat(self,
                        center_heatmap_preds: List[Tensor],
                        wh_preds: List[Tensor],
                        offset_preds: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        rescale: bool = True,
                        with_nms: bool = False) -> InstanceList:
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): WH predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            batch_img_metas (list[dict], optional): Batch image meta info.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to True.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        result_list = []
        for img_id in range(len(batch_img_metas)):
            result_list.append(
                self._predict_by_feat_single(
                    center_heatmap_preds[0][img_id:img_id + 1, ...],
                    wh_preds[0][img_id:img_id + 1, ...],
                    offset_preds[0][img_id:img_id + 1, ...],
                    batch_img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))
        return result_list

    def _predict_by_feat_single(self,
                                center_heatmap_pred: Tensor,
                                wh_pred: Tensor,
                                offset_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True,
                                with_nms: bool = False) -> InstanceData:
        """Transform outputs of a single image into bbox results.

        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            wh_pred (Tensor): WH heatmap for current level with shape
                (1, num_classes, H, W).
            offset_pred (Tensor): Offset for current level with shape
                (1, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Defaults to True.
            with_nms (bool): If True, do nms before return boxes.
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
        batch_det_bboxes, batch_labels = self._decode_heatmap(
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            img_meta['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)

        batch_border = det_bboxes.new_tensor(img_meta['border'])[...,
                                                                 [2, 0, 2, 0]]
        det_bboxes[..., :4] -= batch_border

        if rescale and 'scale_factor' in img_meta:
            det_bboxes[..., :4] /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        if with_nms:
            det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels,
                                                      self.test_cfg)
        results = InstanceData()
        results.bboxes = det_bboxes[..., :4]
        results.scores = det_bboxes[..., 4]
        results.labels = det_labels
        return results

    def _decode_heatmap(self,
                        center_heatmap_pred: Tensor,
                        wh_pred: Tensor,
                        offset_pred: Tensor,
                        img_shape: tuple,
                        k: int = 100,
                        kernel: int = 3) -> Tuple[Tensor, Tensor]:
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (tuple): image shape in hw format.
            k (int): Get top k center keypoints from heatmap. Defaults to 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Defaults to 3.

        Returns:
            tuple[Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes: Tensor, labels: Tensor,
                    cfg: ConfigDict) -> Tuple[Tensor, Tensor]:
        """bboxes nms."""
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:,
                                                             -1].contiguous(),
                                       labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]

        return bboxes, labels
