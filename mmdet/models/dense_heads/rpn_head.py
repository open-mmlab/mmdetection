import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms

from ..builder import HEADS
from .anchor_head import AnchorHead
from .rpn_test_mixin import RPNTestMixin


@HEADS.register_module()
class RPNHead(RPNTestMixin, AnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(1, in_channels, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super(RPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    mlvl_anchors,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                (height, width, 3).
            scale_factors (list[ndarray]): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        batch_size = cls_scores[0].shape[0]
        nms_pre_tensor = torch.tensor(
            cfg.nms_pre, device=cls_scores[0].device, dtype=torch.long)
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(batch_size, -1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(batch_size, -1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(-1)[..., 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).reshape(
                batch_size, -1, 4)
            anchors = mlvl_anchors[idx]
            anchors = anchors.expand_as(rpn_bbox_pred)
            if nms_pre_tensor > 0:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                # keep topk op for dynamic k in onnx model
                if torch.onnx.is_in_onnx_export():
                    # sort op will be converted to TopK in onnx
                    # and k<=3480 in TensorRT
                    scores_shape = torch._shape_as_tensor(scores)
                    nms_pre = torch.where(scores_shape[1] < nms_pre_tensor,
                                          scores_shape[1], nms_pre_tensor)
                    _, topk_inds = scores.topk(nms_pre)
                    batch_inds = torch.arange(batch_size).view(
                        -1, 1).expand_as(topk_inds)
                    scores = scores[batch_inds, topk_inds]
                    rpn_bbox_pred = rpn_bbox_pred[batch_inds, topk_inds, :]
                    anchors = anchors[batch_inds, topk_inds, :]

                elif scores.shape[-1] > cfg.nms_pre:
                    ranked_scores, rank_inds = scores.sort(descending=True)
                    topk_inds = rank_inds[:, :cfg.nms_pre]
                    scores = ranked_scores[:, :cfg.nms_pre]
                    batch_inds = torch.arange(batch_size).view(
                        -1, 1).expand_as(topk_inds)
                    rpn_bbox_pred = rpn_bbox_pred[batch_inds, topk_inds, :]
                    anchors = anchors[batch_inds, topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((
                    batch_size,
                    scores.size(1),
                ),
                                idx,
                                dtype=torch.long))

        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_anchors = torch.cat(mlvl_valid_anchors, dim=1)
        batch_mlvl_rpn_bbox_pred = torch.cat(mlvl_bbox_preds, dim=1)
        batch_mlvl_proposals = self.bbox_coder.decode(
            batch_mlvl_anchors, batch_mlvl_rpn_bbox_pred, max_shape=img_shapes)
        batch_mlvl_ids = torch.cat(level_ids, dim=1)

        # deprecate arguments warning
        if 'nms' not in cfg or 'max_num' in cfg or 'nms_thr' in cfg:
            warnings.warn(
                'In rpn_proposal or test_cfg, '
                'nms_thr has been moved to a dict named nms as '
                'iou_threshold, max_num has been renamed as max_per_img, '
                'name of original arguments and the way to specify '
                'iou_threshold of NMS will be deprecated.')
        if 'nms' not in cfg:
            cfg.nms = ConfigDict(dict(type='nms', iou_threshold=cfg.nms_thr))
        if 'max_num' in cfg:
            if 'max_per_img' in cfg:
                assert cfg.max_num == cfg.max_per_img, f'You ' \
                    f'set max_num and ' \
                    f'max_per_img at the same time, but get {cfg.max_num} ' \
                    f'and {cfg.max_per_img} respectively' \
                    'Please delete max_num which will be deprecated.'
            else:
                cfg.max_per_img = cfg.max_num
        if 'nms_thr' in cfg:
            assert cfg.nms.iou_threshold == cfg.nms_thr, f'You set' \
                f' iou_threshold in nms and ' \
                f'nms_thr at the same time, but get' \
                f' {cfg.nms.iou_threshold} and {cfg.nms_thr}' \
                f' respectively. Please delete the nms_thr ' \
                f'which will be deprecated.'

        result_list = []
        for (mlvl_proposals, mlvl_scores,
             mlvl_ids) in zip(batch_mlvl_proposals, batch_mlvl_scores,
                              batch_mlvl_ids):
            # Skip nonzero op while exporting to ONNX
            if cfg.min_bbox_size > 0 and (not torch.onnx.is_in_onnx_export()):
                w = mlvl_proposals[:, 2] - mlvl_proposals[:, 0]
                h = mlvl_proposals[:, 3] - mlvl_proposals[:, 1]
                valid_ind = torch.nonzero(
                    (w >= cfg.min_bbox_size)
                    & (h >= cfg.min_bbox_size),
                    as_tuple=False).squeeze()
                if valid_ind.sum().item() != len(mlvl_proposals):
                    mlvl_proposals = mlvl_proposals[valid_ind, :]
                    mlvl_scores = mlvl_scores[valid_ind]
                    mlvl_ids = mlvl_ids[valid_ind]

            dets, keep = batched_nms(mlvl_proposals, mlvl_scores, mlvl_ids,
                                     cfg.nms)
            result_list.append(dets[:cfg.max_per_img])
        return result_list
