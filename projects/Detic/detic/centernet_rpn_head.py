# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.dense_heads import CenterNetUpdateHead
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS

INF = 1000000000
RangeType = Sequence[Tuple[int, int]]


@MODELS.register_module(force=True)  # avoid bug
class CenterNetRPNHead(CenterNetUpdateHead):
    """CenterNetUpdateHead is an improved version of CenterNet in CenterNet2.

    Paper link `<https://arxiv.org/abs/2103.07461>`_.
    """

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self._init_reg_convs()
        self._init_predictor()

    def _init_predictor(self) -> None:
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.num_classes, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of each level outputs.

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each \
            scale level, each is a 4D-tensor, the channel number is 4.
        """
        res = multi_apply(self.forward_single, x, self.scales, self.strides)
        return res

    def forward_single(self, x: Tensor, scale: Scale,
                       stride: int) -> Tuple[Tensor, Tensor]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj:`mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps.

        Returns:
            tuple: scores for each class, bbox predictions of
            input feature maps.
        """
        for m in self.reg_convs:
            x = m(x)
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        # bbox_pred needed for gradient computation has been modified
        # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
        # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
        bbox_pred = bbox_pred.clamp(min=0)
        if not self.training:
            bbox_pred *= stride
        return cls_score, bbox_pred  # score aligned, box larger

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
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

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
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []

        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            heatmap = cls_score.sigmoid()
            score_thr = cfg.get('score_thr', 0)

            candidate_inds = heatmap > score_thr  # 0.05
            pre_nms_top_n = candidate_inds.sum()  # N
            pre_nms_top_n = pre_nms_top_n.clamp(max=nms_pre)  # N

            heatmap = heatmap[candidate_inds]  # n

            candidate_nonzeros = candidate_inds.nonzero()  # n
            box_loc = candidate_nonzeros[:, 0]  # n
            labels = candidate_nonzeros[:, 1]  # n

            bbox_pred = bbox_pred[box_loc]  # n x 4
            per_grids = priors[box_loc]  # n x 2

            if candidate_inds.sum().item() > pre_nms_top_n.item():
                heatmap, top_k_indices = \
                    heatmap.topk(pre_nms_top_n, sorted=False)
                labels = labels[top_k_indices]
                bbox_pred = bbox_pred[top_k_indices]
                per_grids = per_grids[top_k_indices]

            bboxes = self.bbox_coder.decode(per_grids, bbox_pred)
            # avoid invalid boxes in RoI heads
            bboxes[:, 2] = torch.max(bboxes[:, 2], bboxes[:, 0] + 0.01)
            bboxes[:, 3] = torch.max(bboxes[:, 3], bboxes[:, 1] + 0.01)

            mlvl_bbox_preds.append(bboxes)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(torch.sqrt(heatmap))
            mlvl_labels.append(labels)

        results = InstanceData()
        results.bboxes = torch.cat(mlvl_bbox_preds)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
