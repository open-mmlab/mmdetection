# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms

from ..builder import HEADS
from .anchor_head import AnchorHead


@HEADS.register_module()
class RPNHead(AnchorHead):
    """RPN head.

    Args:
        in_channels (int): 输入特征图的通道数.
        init_cfg (dict or list[dict], optional): 初始化配置字典.
        num_convs (int): 卷积层数. Default 1.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 num_convs=1,
                 **kwargs):
        self.num_convs = num_convs
        super(RPNHead, self).__init__(
            1, in_channels, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4,
                                 1)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=False)
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
            cls_scores (list[Tensor]): 网络所有层级输出的box_score
                [[bs, na * nc, h, w], ] * num_level
            bbox_preds (list[Tensor]): 网络所有层级输出的box_reg
                [[bs, na * 4, h, w], ] * num_level
            gt_bboxes (list[Tensor]): batch幅图像的gt box
                [[num_gts, 4],] * bs 格式为[x1, y1, x2, y2].
            img_metas (list[dict]): batch幅图像的元信息
            gt_bboxes_ignore (None | list[Tensor]): 计算loss时可以忽略的gt box.
                如果不为None,其shape为[[num_ignored_gts, 4], ] * bs

        Returns:
            dict[str, Tensor]: 计算出的loss字典.
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

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_anchors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """将单张图像的网络输出转换为 roi.

        Args:
            cls_score_list (list[Tensor]): 所有层级的cls_scores,
                    [[na*nc, h, w], ] * num_level
            bbox_pred_list (list[Tensor]): 所有层级的box_reg,
                    [[na*4, h, w], ] * num_level
            score_factor_list (list[Tensor]): 所有层级的box_conf. RPN中为None
            mlvl_anchors (list[Tensor]):所有层级的基础anchor,
                [[h * w * na, 4],] * num_level.
            img_meta (dict): 图像元信息.
            cfg (mmcv.Config): 测试 / 后处理配置,如果为None, test_cfg将被适用.
            rescale (bool): 如果True, 将box缩放回Resize前尺寸空间. 默认: False.
            with_nms (bool): 是否在返回box之前执行nms. 默认: True.

        Returns:
            Tensor: RPN提供的roi [n, 5], (x1, y1, x2, y2, score) .
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']

        # level_ids 用作 batch-NMS 的label以将它们分开,可能是为了增加roi数量.
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # 在RPN中, 前景∈[0, nc) 背景=nc
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            anchors = mlvl_anchors[level_idx]
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
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        return self._bbox_post_process(mlvl_scores, mlvl_bbox_preds,
                                       mlvl_valid_anchors, level_ids, cfg,
                                       img_shape)

    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors,
                           level_ids, cfg, img_shape, **kwargs):
        """box后处理方法.

        Args:
            mlvl_scores (list[Tensor]): 所有层级上的box score,
                [[nms_pre, ],] * num_level
            mlvl_bboxes (list[Tensor]): 所有层级上的box reg,
                [[nms_pre, 4],] * num_level.
            mlvl_valid_anchors (list[Tensor]): 所有层级上的基础anchor,
                [[nms_pre, 4],] * num_level.
            level_ids (list[Tensor]): 当前anchor/reg/score所属的层级索引
                [[nms_pre, ],] * num_level
            cfg (mmcv.Config): 测试 / 后处理配置,如果为None, test_cfg将被适用.
            img_shape (tuple(int)): 图像Resize后的尺寸,也即batch幅图片对齐前的尺寸.

        Returns:
            Tensor: RPN提供的roi [max_per_img, 5], (x1, y1, x2, y2, score) ..
        """
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() > 0:
            dets, _ = batched_nms(proposals, scores, ids, cfg.nms)
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg.max_per_img]

    def onnx_export(self, x, img_metas):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.
        Returns:
            Tensor: dets of shape [N, num_det, 5].
        """
        cls_scores, bbox_preds = self(x)

        assert len(cls_scores) == len(bbox_preds)

        batch_bboxes, batch_scores = super(RPNHead, self).onnx_export(
            cls_scores, bbox_preds, img_metas=img_metas, with_nms=False)
        # Use ONNX::NonMaxSuppression in deployment
        from mmdet.core.export import add_dummy_nms_for_onnx
        cfg = copy.deepcopy(self.test_cfg)
        score_threshold = cfg.nms.get('score_thr', 0.0)
        nms_pre = cfg.get('deploy_nms_pre', -1)
        # Different from the normal forward doing NMS level by level,
        # we do NMS across all levels when exporting ONNX.
        dets, _ = add_dummy_nms_for_onnx(batch_bboxes, batch_scores,
                                         cfg.max_per_img,
                                         cfg.nms.iou_threshold,
                                         score_threshold, nms_pre,
                                         cfg.max_per_img)
        return dets
