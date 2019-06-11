import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmdet.core import multi_apply, multiclass_nms, distance2bbox

from ..losses import sigmoid_focal_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule


def select_iou_loss(pred, target, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = pred.size(0)
    assert pred.size(0) == target.size(0)
    target = target.clamp(min=0.)
    area_pred = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
    area_gt = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])
    area_i = ((torch.min(pred[:, 0], target[:, 0]) +
               torch.min(pred[:, 2], target[:, 2])) *
              (torch.min(pred[:, 1], target[:, 1]) +
               torch.min(pred[:, 3], target[:, 3])))
    area_u = area_pred + area_gt - area_i
    iou = area_i / area_u
    loc_losses = -torch.log(iou.clamp(min=1e-7))
    return torch.sum(weight * loc_losses) / avg_factor


@HEADS.register_module
class FSAFHead(nn.Module):
    """Feature Selective Anchor-Free Head

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        stacked_convs (int): Number of conv layers before head.
        norm_factor (float): Distance normalization factor.
        feat_strides (Iterable): Feature strides.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 norm_factor=4.0,
                 feat_strides=[8, 16, 32, 64, 128],
                 conv_cfg=None,
                 norm_cfg=None):
        super(FSAFHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.norm_factor = norm_factor
        self.feat_strides = feat_strides
        self.cls_out_channels = self.num_classes - 1
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self):
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
        self.fsaf_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fsaf_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fsaf_cls, std=0.01, bias=bias_cls)
        normal_init(self.fsaf_reg, std=0.01, bias=0.1)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.fsaf_cls(cls_feat)
        bbox_pred = self.relu(self.fsaf_reg(reg_feat))
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_locs, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = sigmoid_focal_loss(
            cls_score,
            labels,
            weight=label_weights,
            gamma=cfg.gamma,
            alpha=cfg.alpha,
            avg_factor=num_total_samples)
        # localization loss
        if bbox_targets.size(0) == 0:
            loss_bbox = bbox_pred.new_zeros(1)
        else:
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred[bbox_locs[:, 0], bbox_locs[:, 1],
                                  bbox_locs[:, 2], :]
            loss_bbox = select_iou_loss(
                bbox_pred,
                bbox_targets,
                cfg.bbox_reg_weight,
                avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        cls_reg_targets = self.point_target(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            img_metas,
            cfg,
            gt_labels_list=gt_labels,
            gt_bboxes_ignore_list=gt_bboxes_ignore)
        # if cls_reg_targets is None:
        #     return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_locs_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = num_total_pos
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_locs_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def point_target(self,
                     cls_scores,
                     bbox_preds,
                     gt_bboxes,
                     img_metas,
                     cfg,
                     gt_labels_list=None,
                     gt_bboxes_ignore_list=None):
        num_imgs = len(img_metas)
        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        # split net outputs w.r.t. images
        num_levels = len(self.feat_strides)
        assert len(cls_scores) == len(bbox_preds) == num_levels
        cls_score_list = []
        bbox_pred_list = []
        for img_id in range(num_imgs):
            cls_score_list.append(
                [cls_scores[i][img_id].detach() for i in range(num_levels)])
            bbox_pred_list.append(
                [bbox_preds[i][img_id].detach() for i in range(num_levels)])

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_locs,
         num_pos_list, num_neg_list) = multi_apply(
             self.point_target_single,
             cls_score_list,
             bbox_pred_list,
             gt_bboxes,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             cfg=cfg)
        # correct image index in bbox_locs
        for i in range(num_imgs):
            for lvl in range(num_levels):
                all_bbox_locs[i][lvl][:, 0] = i

        # sampled points of all images
        num_total_pos = sum([max(num, 1) for num in num_pos_list])
        num_total_neg = sum([max(num, 1) for num in num_neg_list])
        # combine targets to a list w.r.t. multiple levels
        labels_list = self.images_to_levels(all_labels, num_imgs, num_levels,
                                            True)
        label_weights_list = self.images_to_levels(all_label_weights, num_imgs,
                                                   num_levels, True)
        bbox_targets_list = self.images_to_levels(all_bbox_targets, num_imgs,
                                                  num_levels, False)
        bbox_locs_list = self.images_to_levels(all_bbox_locs, num_imgs,
                                               num_levels, False)
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_locs_list, num_total_pos, num_total_neg)

    def point_target_single(self, cls_score_list, bbox_pred_list, gt_bboxes,
                            gt_bboxes_ignore, gt_labels, img_meta, cfg):
        num_levels = len(self.feat_strides)
        assert len(cls_score_list) == len(bbox_pred_list) == num_levels
        feat_lvls = self.feat_level_select(cls_score_list, bbox_pred_list,
                                           gt_bboxes, gt_labels, cfg)
        labels = []
        label_weights = []
        bbox_targets = []
        bbox_locs = []
        device = bbox_pred_list[0].device
        img_h, img_w, _ = img_meta['pad_shape']
        for lvl in range(num_levels):
            stride = self.feat_strides[lvl]
            norm = stride * self.norm_factor
            inds = torch.nonzero(feat_lvls == lvl).squeeze(-1)
            h, w = cls_score_list[lvl].size()[-2:]
            valid_h = min(int(np.ceil(img_h / stride)), h)
            valid_w = min(int(np.ceil(img_w / stride)), w)

            _labels = torch.zeros_like(
                cls_score_list[lvl][0], dtype=torch.long)
            _label_weights = torch.zeros_like(
                cls_score_list[lvl][0], dtype=torch.float)
            _label_weights[:valid_h, :valid_w] = 1.
            _bbox_targets = bbox_pred_list[lvl].new_zeros((0, 4),
                                                          dtype=torch.float)
            _bbox_locs = bbox_pred_list[lvl].new_zeros((0, 3),
                                                       dtype=torch.long)

            if len(inds) > 0:
                boxes = gt_bboxes[inds, :]
                classes = gt_labels[inds]
                proj_boxes = boxes / stride
                ig_x1, ig_y1, ig_x2, ig_y2 = self.prop_box_bounds(
                    proj_boxes, cfg.ignore_scale, w, h)
                pos_x1, pos_y1, pos_x2, pos_y2 = self.prop_box_bounds(
                    proj_boxes, cfg.pos_scale, w, h)
                for i in range(len(inds)):
                    # setup classification ground-truth
                    _labels[pos_y1[i]:pos_y2[i], pos_x1[i]:
                            pos_x2[i]] = classes[i]
                    _label_weights[ig_y1[i]:ig_y2[i], ig_x1[i]:ig_x2[i]] = 0.
                    _label_weights[pos_y1[i]:pos_y2[i], pos_x1[i]:
                                   pos_x2[i]] = 1.
                    # setup localization ground-truth
                    locs_x = torch.arange(
                        pos_x1[i], pos_x2[i], device=device, dtype=torch.long)
                    locs_y = torch.arange(
                        pos_y1[i], pos_y2[i], device=device, dtype=torch.long)
                    shift_x = (locs_x.float() + 0.5) * stride
                    shift_y = (locs_y.float() + 0.5) * stride
                    shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
                    shifts = torch.stack(
                        (shift_xx, shift_yy, shift_xx, shift_yy), dim=-1)
                    shifts[:, 0] = shifts[:, 0] - boxes[i, 0]
                    shifts[:, 1] = shifts[:, 1] - boxes[i, 1]
                    shifts[:, 2] = boxes[i, 2] - shifts[:, 2]
                    shifts[:, 3] = boxes[i, 3] - shifts[:, 3]
                    _bbox_targets = torch.cat((_bbox_targets, shifts / norm),
                                              dim=0)
                    locs_xx, locs_yy = self._meshgrid(locs_x, locs_y)
                    zeros = torch.zeros_like(locs_xx)
                    locs = torch.stack((zeros, locs_yy, locs_xx), dim=-1)
                    _bbox_locs = torch.cat((_bbox_locs, locs), dim=0)

            labels.append(_labels)
            label_weights.append(_label_weights)
            bbox_targets.append(_bbox_targets)
            bbox_locs.append(_bbox_locs)

        # ignore regions in adjacent pyramids
        for lvl in range(num_levels):
            stride = self.feat_strides[lvl]
            w, h = cls_score_list[lvl].size()[-2:]
            # lower pyramid if exists
            if lvl > 0:
                inds = torch.nonzero(feat_lvls == lvl - 1).squeeze(-1)
                if len(inds) > 0:
                    boxes = gt_bboxes[inds, :]
                    proj_boxes = boxes / stride
                    ig_x1, ig_y1, ig_x2, ig_y2 = self.prop_box_bounds(
                        proj_boxes, cfg.ignore_scale, w, h)
                    for i in range(len(inds)):
                        label_weights[lvl][ig_y1[i]:ig_y2[i], ig_x1[i]:
                                           ig_x2[i]] = 0.
            # upper pyramid if exists
            if lvl < num_levels - 1:
                inds = torch.nonzero(feat_lvls == lvl + 1).squeeze(-1)
                if len(inds) > 0:
                    boxes = gt_bboxes[inds, :]
                    proj_boxes = boxes / stride
                    ig_x1, ig_y1, ig_x2, ig_y2 = self.prop_box_bounds(
                        proj_boxes, cfg.ignore_scale, w, h)
                    for i in range(len(inds)):
                        label_weights[lvl][ig_y1[i]:ig_y2[i], ig_x1[i]:
                                           ig_x2[i]] = 0.

        # compute number of foreground and background points
        num_pos = 0
        num_neg = 0
        for lvl in range(num_levels):
            npos = bbox_targets[lvl].size(0)
            num_pos += npos
            num_neg += (label_weights[lvl].nonzero().size(0) - npos)
        return (labels, label_weights, bbox_targets, bbox_locs, num_pos,
                num_neg)

    def feat_level_select(self, cls_score_list, bbox_pred_list, gt_bboxes,
                          gt_labels, cfg):
        if cfg.online_select:
            num_levels = len(cls_score_list)
            num_boxes = gt_bboxes.size(0)
            feat_losses = gt_bboxes.new_zeros((num_boxes, num_levels))
            device = bbox_pred_list[0].device
            for lvl in range(num_levels):
                stride = self.feat_strides[lvl]
                norm = stride * self.norm_factor
                cls_score = cls_score_list[lvl].permute(1, 2, 0)  # h x w x C
                bbox_pred = bbox_pred_list[lvl].permute(1, 2, 0)  # h x w x 4
                h, w = cls_score.size()[:2]

                proj_boxes = gt_bboxes / stride
                x1, y1, x2, y2 = self.prop_box_bounds(proj_boxes,
                                                      cfg.pos_scale, w, h)

                for i in range(num_boxes):
                    locs_x = torch.arange(
                        x1[i], x2[i], device=device, dtype=torch.long)
                    locs_y = torch.arange(
                        y1[i], y2[i], device=device, dtype=torch.long)
                    locs_xx, locs_yy = self._meshgrid(locs_x, locs_y)
                    avg_factor = locs_xx.size(0)
                    # classification focal loss
                    scores = cls_score[locs_yy, locs_xx, :]
                    labels = gt_labels[i].repeat(avg_factor)
                    label_weights = torch.ones_like(labels).float()
                    loss_cls = sigmoid_focal_loss(
                        scores,
                        labels,
                        weight=label_weights,
                        gamma=cfg.gamma,
                        alpha=cfg.alpha,
                        avg_factor=avg_factor)
                    # localization iou loss
                    deltas = bbox_pred[locs_yy, locs_xx, :]
                    shift_x = (locs_x.float() + 0.5) * stride
                    shift_y = (locs_y.float() + 0.5) * stride
                    shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
                    shifts = torch.stack(
                        (shift_xx, shift_yy, shift_xx, shift_yy), dim=-1)
                    shifts[:, 0] = shifts[:, 0] - gt_bboxes[i, 0]
                    shifts[:, 1] = shifts[:, 1] - gt_bboxes[i, 1]
                    shifts[:, 2] = gt_bboxes[i, 2] - shifts[:, 2]
                    shifts[:, 3] = gt_bboxes[i, 3] - shifts[:, 3]
                    loss_loc = select_iou_loss(deltas, shifts / norm,
                                               cfg.bbox_reg_weight, avg_factor)
                    feat_losses[i, lvl] = loss_cls + loss_loc
            feat_levels = torch.argmin(feat_losses, dim=1)
        else:
            num_levels = len(self.feat_strides)
            lvl0 = cfg.canonical_level
            s0 = cfg.canonical_scale
            assert 0 <= lvl0 < num_levels
            gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
            s = torch.sqrt(gt_w * gt_h)
            # FPN Eq. (1)
            feat_levels = torch.floor(lvl0 + torch.log2(s / s0 + 1e-6))
            feat_levels = torch.clamp(feat_levels, 0, num_levels - 1).int()
        return feat_levels

    def xyxy2xcycwh(self, xyxy):
        """Convert [x1 y1 x2 y2] box format to [xc yc w h] format."""
        return torch.cat(
            (0.5 * (xyxy[:, 0:2] + xyxy[:, 2:4]), xyxy[:, 2:4] - xyxy[:, 0:2]),
            dim=1)

    def xcycwh2xyxy(self, xywh):
        """Convert [xc yc w y] box format to [x1 y1 x2 y2] format."""
        return torch.cat((xywh[:, 0:2] - 0.5 * xywh[:, 2:4],
                          xywh[:, 0:2] + 0.5 * xywh[:, 2:4]),
                         dim=1)

    def prop_box_bounds(self, boxes, scale, width, height):
        """Compute proportional box regions.

        Box centers are fixed. Box w and h scaled by scale.
        """
        prop_boxes = self.xyxy2xcycwh(boxes)
        prop_boxes[:, 2:] *= scale
        prop_boxes = self.xcycwh2xyxy(prop_boxes)
        x1 = torch.floor(prop_boxes[:, 0]).clamp(0, width - 1).int()
        y1 = torch.floor(prop_boxes[:, 1]).clamp(0, height - 1).int()
        x2 = torch.ceil(prop_boxes[:, 2]).clamp(1, width).int()
        y2 = torch.ceil(prop_boxes[:, 3]).clamp(1, height).int()
        return x1, y1, x2, y2

    def images_to_levels(self, target, num_imgs, num_levels, is_cls=True):
        level_target = []
        if is_cls:
            for lvl in range(num_levels):
                level_target.append(
                    torch.stack([target[i][lvl] for i in range(num_imgs)],
                                dim=0))
        else:
            for lvl in range(num_levels):
                level_target.append(
                    torch.cat([target[j][lvl] for j in range(num_imgs)],
                              dim=0))
        return level_target

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        num_levels = len(self.feat_strides)
        assert len(cls_scores) == len(bbox_preds) == num_levels
        device = bbox_preds[0].device
        dtype = bbox_preds[0].dtype

        mlvl_points = [
            self.generate_points(
                bbox_preds[i].size()[-2:],
                self.feat_strides[i],
                device=device,
                dtype=dtype) for i in range(num_levels)
        ]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() * self.feat_strides[i] *
                self.norm_factor for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_points, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, points in zip(cls_scores, bbox_preds,
                                                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            scores = cls_score.sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                points = points[topk_inds, :]
            bboxes = distance2bbox(points, bbox_pred, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels

    def generate_points(self,
                        featmap_size,
                        stride=16,
                        device='cuda',
                        dtype=torch.float32):
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device, dtype=dtype) + 0.5
        shift_y = torch.arange(0, feat_h, device=device, dtype=dtype) + 0.5
        shift_x *= stride
        shift_y *= stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        points = torch.stack((shift_xx, shift_yy), dim=-1)
        return points

    def _meshgrid(self, x, y):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        return xx, yy
