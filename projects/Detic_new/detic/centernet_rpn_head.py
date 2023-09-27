# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.dense_heads import CenterNetUpdateHead
from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2distance
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from .iou_loss import IOULoss

# from .heatmap_focal_loss import binary_heatmap_focal_loss_jit
INF = 1000000000
RangeType = Sequence[Tuple[int, int]]


@MODELS.register_module()
class CenterNetRPNHead(CenterNetUpdateHead):
    """CenterNetUpdateHead is an improved version of CenterNet in CenterNet2.

    Paper link `<https://arxiv.org/abs/2103.07461>`_.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channel in the input feature map.
        regress_ranges (Sequence[Tuple[int, int]]): Regress range of multiple
            level points.
        hm_min_radius (int): Heatmap target minimum radius of cls branch.
            Defaults to 4.
        hm_min_overlap (float): Heatmap target minimum overlap of cls branch.
            Defaults to 0.8.
        more_pos_thresh (float): The filtering threshold when the cls branch
            adds more positive samples. Defaults to 0.2.
        more_pos_topk (int): The maximum number of additional positive samples
            added to each gt. Defaults to 9.
        soft_weight_on_reg (bool): Whether to use the soft target of the
            cls branch as the soft weight of the bbox branch.
            Defaults to False.
        loss_cls (:obj:`ConfigDict` or dict): Config of cls loss. Defaults to
            dict(type='GaussianFocalLoss', loss_weight=1.0)
        loss_bbox (:obj:`ConfigDict` or dict): Config of bbox loss. Defaults to
             dict(type='GIoULoss', loss_weight=2.0).
        norm_cfg (:obj:`ConfigDict` or dict, optional): dictionary to construct
            and config norm layer.  Defaults to
            ``norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)``.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config.
            Unused in CenterNet. Reserved for compatibility with
            SingleStageDetector.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config
            of CenterNet.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 regress_ranges: RangeType = ((0, 80), (64, 160), (128, 320),
                                              (256, 640), (512, INF)),
                 hm_min_radius: int = 4,
                 hm_min_overlap: float = 0.8,
                 more_pos: bool = False,
                 more_pos_thresh: float = 0.2,
                 more_pos_topk: int = 9,
                 soft_weight_on_reg: bool = False,
                 not_clamp_box: bool = False,
                 loss_cls: ConfigType = dict(
                     type='HeatmapFocalLoss',
                     alpha=0.25,
                     beta=4.0,
                     gamma=2.0,
                     pos_weight=1.0,
                     neg_weight=1.0,
                     sigmoid_clamp=1e-4,
                     ignore_high_fp=-1.0,
                     loss_weight=1.0,
                 ),
                 loss_bbox: ConfigType = dict(
                     type='GIoULoss', loss_weight=2.0),
                 norm_cfg: OptConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            # loss_bbox=loss_bbox,
            loss_cls=loss_cls,
            norm_cfg=norm_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)
        self.soft_weight_on_reg = soft_weight_on_reg
        self.hm_min_radius = hm_min_radius
        self.more_pos_thresh = more_pos_thresh
        self.more_pos_topk = more_pos_topk
        self.more_pos = more_pos
        self.not_clamp_box = not_clamp_box
        self.delta = (1 - hm_min_overlap) / (1 + hm_min_overlap)
        self.loss_bbox = IOULoss('giou')

        # GaussianFocalLoss must be sigmoid mode
        self.use_sigmoid_cls = True
        self.cls_out_channels = num_classes

        self.regress_ranges = regress_ranges
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self._init_reg_convs()
        self._init_predictor()

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
        return cls_score, bbox_pred  # score aligned, box larger

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is 4.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
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

        num_imgs = cls_scores[0].size(0)
        assert len(cls_scores) == len(bbox_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        # 1 flatten outputs
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        assert (torch.isfinite(flatten_bbox_preds).all().item())

        # 2 calc reg and cls branch targets
        cls_targets, bbox_targets = self.get_targets(all_level_points,
                                                     batch_gt_instances)

        # 3 pos index for cls branch
        featmap_sizes = flatten_points.new_tensor(featmap_sizes)

        if self.more_pos:
            pos_inds, cls_labels = self.add_cls_pos_inds(
                flatten_points, flatten_bbox_preds, featmap_sizes,
                batch_gt_instances)
        else:
            pos_inds = self._get_label_inds(batch_gt_instances,
                                            batch_img_metas, featmap_sizes)

        # 4 calc cls loss
        if pos_inds is None:
            # num_gts=0
            num_pos_cls = bbox_preds[0].new_tensor(0, dtype=torch.float)
        else:
            num_pos_cls = bbox_preds[0].new_tensor(
                len(pos_inds), dtype=torch.float)
        num_pos_cls = max(reduce_mean(num_pos_cls), 1.0)

        cat_agn_cls_targets = cls_targets.max(dim=1)[0]  # M

        cls_pos_loss, cls_neg_loss = self.loss_cls(
            flatten_cls_scores.squeeze(1), cat_agn_cls_targets, pos_inds,
            num_pos_cls)

        # 5 calc reg loss
        pos_bbox_inds = torch.nonzero(
            bbox_targets.max(dim=1)[0] >= 0).squeeze(1)
        pos_bbox_preds = flatten_bbox_preds[pos_bbox_inds]
        pos_bbox_targets = bbox_targets[pos_bbox_inds]

        bbox_weight_map = cls_targets.max(dim=1)[0]
        bbox_weight_map = bbox_weight_map[pos_bbox_inds]
        bbox_weight_map = bbox_weight_map if self.soft_weight_on_reg \
            else torch.ones_like(bbox_weight_map)

        num_pos_bbox = max(reduce_mean(bbox_weight_map.sum()), 1.0)

        if len(pos_bbox_inds) > 0:
            bbox_loss = self.loss_bbox(
                pos_bbox_preds,
                pos_bbox_targets,
                bbox_weight_map,
                reduction='sum') / num_pos_bbox
        else:
            bbox_loss = flatten_bbox_preds.sum() * 0

        return dict(
            loss_bbox=bbox_loss,
            loss_cls_pos=cls_pos_loss,
            loss_cls_neg=cls_neg_loss)

    def loss_and_predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
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
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(x)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg)
        return losses, predictions

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

            bbox_pred = bbox_pred * self.strides[level_idx]

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

            bboxes = torch.stack([
                per_grids[:, 0] - bbox_pred[:, 0],
                per_grids[:, 1] - bbox_pred[:, 1],
                per_grids[:, 0] + bbox_pred[:, 2],
                per_grids[:, 1] + bbox_pred[:, 3],
            ],
                                 dim=1)  # n x 4

            # avoid invalid boxes in RoI heads
            bboxes[:, 2] = torch.max(bboxes[:, 2], bboxes[:, 0] + 0.01)
            bboxes[:, 3] = torch.max(bboxes[:, 3], bboxes[:, 1] + 0.01)

            # bboxes = self.bbox_coder.decode(per_grids, bbox_pred)
            # # avoid invalid boxes in RoI heads
            # bboxes[:, 2] = torch.max(bboxes[:, 2], bboxes[:, 0] + 0.01)
            # bboxes[:, 3] = torch.max(bboxes[:, 3], bboxes[:, 1] + 0.01)

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

    def _get_label_inds(self, batch_gt_instances, batch_img_metas,
                        shapes_per_level):
        '''
        Inputs:
            batch_gt_instances: [n_i], sum n_i = N
            shapes_per_level: L x 2 [(h_l, w_l)]_L
        Returns:
            pos_inds: N'
            labels: N'
        '''
        pos_inds = []
        L = len(self.strides)
        B = len(batch_gt_instances)
        shapes_per_level = shapes_per_level.long()
        loc_per_level = (shapes_per_level[:, 0] *
                         shapes_per_level[:, 1]).long()  # L
        level_bases = []
        s = 0
        for i in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[i]
        level_bases = shapes_per_level.new_tensor(level_bases).long()  # L
        strides_default = shapes_per_level.new_tensor(
            self.strides).float()  # L
        for im_i in range(B):
            targets_per_im = batch_gt_instances[im_i]
            if hasattr(targets_per_im, 'bboxes'):
                bboxes = targets_per_im.bboxes  # n x 4
            else:
                bboxes = targets_per_im.labels.new_tensor(
                    [], dtype=torch.float).reshape(-1, 4)
            n = bboxes.shape[0]
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2)  # n x 2
            centers = centers.view(n, 1, 2).expand(n, L, 2).contiguous()
            if self.not_clamp_box:
                h, w = batch_img_metas[im_i]._image_size
                centers[:, :, 0].clamp_(min=0).clamp_(max=w - 1)
                centers[:, :, 1].clamp_(min=0).clamp_(max=h - 1)
            strides = strides_default.view(1, L, 1).expand(n, L, 2)
            centers_inds = (centers / strides).long()  # n x L x 2
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            pos_ind = level_bases.view(1, L).expand(n, L) \
                + im_i * loc_per_level.view(1, L).expand(n, L) \
                + centers_inds[:, :, 1] * Ws + centers_inds[:, :, 0]  # n x L
            is_cared_in_the_level = self.assign_fpn_level(bboxes)
            pos_ind = pos_ind[is_cared_in_the_level].view(-1)

            pos_inds.append(pos_ind)  # n'
        pos_inds = torch.cat(pos_inds, dim=0).long()
        return pos_inds  # N, N

    def assign_fpn_level(self, boxes):
        '''
        Inputs:
            boxes: n x 4
            size_ranges: L x 2
        Return:
            is_cared_in_the_level: n x L
        '''
        size_ranges = boxes.new_tensor(self.regress_ranges).view(
            len(self.regress_ranges), 2)  # L x 2
        crit = ((boxes[:, 2:] - boxes[:, :2])**2).sum(dim=1)**0.5 / 2  # n
        n, L = crit.shape[0], size_ranges.shape[0]
        crit = crit.view(n, 1).expand(n, L)
        size_ranges_expand = size_ranges.view(1, L, 2).expand(n, L, 2)
        is_cared_in_the_level = (crit >= size_ranges_expand[:, :, 0]) & \
            (crit <= size_ranges_expand[:, :, 1])
        return is_cared_in_the_level

    def _get_targets_single(self, gt_instances: InstanceData, points: Tensor,
                            regress_ranges: Tensor,
                            strides: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute classification and bbox targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_labels = gt_instances.labels

        if not hasattr(gt_instances, 'bboxes'):
            gt_bboxes = gt_labels.new_tensor([], dtype=torch.float)
        else:
            gt_bboxes = gt_instances.bboxes

        if not hasattr(gt_instances, 'bboxes') or num_gts == 0:
            return gt_labels.new_full((num_points,
                                       self.num_classes),
                                      self.num_classes,
                                      dtype=torch.float), \
                   gt_bboxes.new_full((num_points, 4), -1)

        # Calculate the regression tblr target corresponding to all points
        points = points[:, None].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        strides = strides[:, None, None].expand(num_points, num_gts, 2)

        bbox_target = bbox2distance(points, gt_bboxes)  # M x N x 4

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_target.min(dim=2)[0] > 0  # M x N

        # condition2: Calculate the nearest points from
        # the upper, lower, left and right ranges from
        # the center of the gt bbox
        centers = ((gt_bboxes[..., [0, 1]] + gt_bboxes[..., [2, 3]]) / 2)
        centers_discret = ((centers / strides).int() * strides).float() + \
            strides / 2

        centers_discret_dist = points - centers_discret
        dist_x = centers_discret_dist[..., 0].abs()
        dist_y = centers_discret_dist[..., 1].abs()
        inside_gt_center3x3_mask = (dist_x <= strides[..., 0]) & \
                                   (dist_y <= strides[..., 0])

        # condition3： limit the regression range for each location
        bbox_target_wh = bbox_target[..., :2] + bbox_target[..., 2:]
        crit = (bbox_target_wh**2).sum(dim=2)**0.5 / 2
        inside_fpn_level_mask = (crit >= regress_ranges[:, [0]]) & \
                                (crit <= regress_ranges[:, [1]])
        bbox_target_mask = inside_gt_bbox_mask & \
            inside_gt_center3x3_mask & \
            inside_fpn_level_mask

        # Calculate the distance weight map
        gt_center_peak_mask = ((centers_discret_dist**2).sum(dim=2) == 0)
        weighted_dist = ((points - centers)**2).sum(dim=2)  # M x N
        weighted_dist[gt_center_peak_mask] = 0

        areas = (gt_bboxes[..., 2] - gt_bboxes[..., 0]) * (
            gt_bboxes[..., 3] - gt_bboxes[..., 1])
        radius = self.delta**2 * 2 * areas
        radius = torch.clamp(radius, min=self.hm_min_radius**2)
        weighted_dist = weighted_dist / radius

        # Calculate bbox_target
        bbox_weighted_dist = weighted_dist.clone()
        bbox_weighted_dist[bbox_target_mask == 0] = INF * 1.0
        min_dist, min_inds = bbox_weighted_dist.min(dim=1)
        bbox_target = bbox_target[range(len(bbox_target)),
                                  min_inds]  # M x N x 4 --> M x 4
        bbox_target[min_dist == INF] = -INF

        # Convert to feature map scale
        bbox_target /= strides[:, 0, :].repeat(1, 2)

        # Calculate cls_target
        cls_target = self._create_heatmaps_from_dist(weighted_dist, gt_labels)

        return cls_target, bbox_target
