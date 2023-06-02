# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import (cat_boxes, empty_box_as, get_box_tensor,
                                   get_box_wh, scale_boxes)
from mmdet.utils import InstanceList, MultiConfig, OptInstanceList
from .anchor_head import AnchorHead


@MODELS.register_module()
class RPNHead(AnchorHead):
    """Implementation of RPN head.

    Args:
<<<<<<< HEAD
        in_channels (int): 输入特征图的通道数.
        init_cfg (dict or list[dict], optional): 初始化配置字典.
        num_convs (int): 卷积层数. Default 1.
=======
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of categories excluding the background
            category. Defaults to 1.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or \
            list[dict]): Initialization config dict.
        num_convs (int): Number of convolution layers in the head.
            Defaults to 1.
>>>>>>> mmdetection/main
    """  # noqa: W605

    def __init__(self,
                 in_channels: int,
                 num_classes: int = 1,
                 init_cfg: MultiConfig = dict(
                     type='Normal', layer='Conv2d', std=0.01),
                 num_convs: int = 1,
                 **kwargs) -> None:
        self.num_convs = num_convs
        assert num_classes == 1
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self) -> None:
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
        reg_dim = self.bbox_coder.encode_size
        self.rpn_reg = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * reg_dim, 1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """
        x = self.rpn_conv(x)
<<<<<<< HEAD
        x = F.relu(x, inplace=False)
=======
        x = F.relu(x)
>>>>>>> mmdetection/main
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) \
            -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
<<<<<<< HEAD
            cls_scores (list[Tensor]): 网络所有层级输出的box_score
                [[bs, na * nc, h, w], ] * num_level
            bbox_preds (list[Tensor]): 网络所有层级输出的box_reg
                [[bs, na * 4, h, w], ] * num_level
            gt_bboxes (list[Tensor]): batch幅图像的gt box
                [[num_gts, 4],] * bs 格式为[x1, y1, x2, y2].
            img_metas (list[dict]): batch幅图像的元信息
            gt_bboxes_ignore (None | list[Tensor]): 计算loss时可以忽略的gt box.
                如果不为None,其shape为[[num_ignored_gts, 4], ] * bs
=======
            cls_scores (list[Tensor]): Box scores for each scale level,
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[obj:InstanceData]): Batch of gt_instance.
                It usually includes ``bboxes`` and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[obj:InstanceData], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
>>>>>>> mmdetection/main

        Returns:
            dict[str, Tensor]: 计算出的loss字典.
        """
        losses = super().loss_by_feat(
            cls_scores,
            bbox_preds,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

<<<<<<< HEAD
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
=======
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
            score_factor_list (list[Tensor]): Be compatible with
                BaseDenseHead. Not used in RPNHead.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (ConfigDict, optional): Test / postprocessing configuration,
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
>>>>>>> mmdetection/main
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
<<<<<<< HEAD

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
=======
        nms_pre = cfg.get('nms_pre', -1)
>>>>>>> mmdetection/main

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        level_ids = []
        for level_idx, (cls_score, bbox_pred, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            reg_dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, reg_dim)
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0] since mmdet v2.0
                # BG cat_id: 1
                scores = cls_score.softmax(-1)[:, :-1]

            scores = torch.squeeze(scores)
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                bbox_pred = bbox_pred[topk_inds, :]
                priors = priors[topk_inds]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)

            # use level id to implement the separate level nms
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

<<<<<<< HEAD
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
=======
        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.level_ids = torch.cat(level_ids)

        return self._bbox_post_process(
            results=results, cfg=cfg, rescale=rescale, img_meta=img_meta)

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.
>>>>>>> mmdetection/main

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert with_nms, '`with_nms` must be True in RPNHead'
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.level_ids, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]
            # TODO: This would unreasonably show the 0th class label
            #  in visualization
            results.labels = results.scores.new_zeros(
                len(results), dtype=torch.long)
            del results.level_ids
        else:
            # To avoid some potential error
            results_ = InstanceData()
            results_.bboxes = empty_box_as(results.bboxes)
            results_.scores = results.scores.new_zeros(0)
            results_.labels = results.scores.new_zeros(0)
            results = results_
        return results
