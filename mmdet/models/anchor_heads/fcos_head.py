import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..utils import bias_init_with_prob, Scale
from ...core.anchor import centerness_target, fcos_target
from ...core.loss import sigmoid_focal_loss, iou_loss
from ...core.post_processing import singleclass_nms
from ...core.utils import multi_apply

INF = 1e8


@HEADS.register_module
class FCOSHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF))):
        super(FCOSHead, self).__init__()

        self.num_classes = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges

        self._init_layers()

    def _init_layers(self):
        self.cls_layers = nn.ModuleList()
        self.reg_layers = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_layers.append(
                nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1))
            self.cls_layers.append(nn.GroupNorm(32, self.feat_channels))
            self.cls_layers.append(nn.ReLU(inplace=False))
            self.reg_layers.append(
                nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1))
            self.reg_layers.append(nn.GroupNorm(32, self.feat_channels))
            self.reg_layers.append(nn.ReLU(inplace=False))
        self.fcos_cls = nn.Conv2d(self.feat_channels, self.num_classes, 3, 1,
                                  1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, 1, 1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, 1, 1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

    def init_weights(self):
        for l in self.cls_layers:
            if isinstance(l, nn.Conv2d):
                normal_init(l, std=0.01)
        for l in self.reg_layers:
            if isinstance(l, nn.Conv2d):
                normal_init(l, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_layers:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)
        centerness = self.fcos_centerness(cls_feat)

        for reg_layer in self.reg_layers:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = torch.exp(scale(self.fcos_reg(reg_feat)))
        return cls_score, bbox_pred, centerness

    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_centers = self.get_centers(featmap_sizes, self.strides,
                                             cls_scores[0].dtype,
                                             cls_scores[0].device)
        labels, bbox_targets = fcos_target(
            all_level_centers, self.regress_ranges, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = sigmoid_focal_loss(
            flatten_cls_scores, flatten_labels, cfg.gamma, cfg.alpha,
            'none').sum()[None] / (num_pos + num_imgs)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_centerness_targets = centerness_target(pos_bbox_targets)

        if num_pos > 0:
            loss_reg = (
                (iou_loss(pos_bbox_preds, pos_bbox_targets, reduction='none') *
                 pos_centerness_targets).sum() /
                pos_centerness_targets.sum())[None]
            loss_centerness = F.binary_cross_entropy_with_logits(
                pos_centerness, pos_centerness_targets, reduction='mean')[None]
        else:
            loss_reg = pos_bbox_preds.sum()[None]
            loss_centerness = pos_centerness.sum()[None]

        return dict(
            loss_cls=loss_cls,
            loss_reg=loss_reg,
            loss_centerness=loss_centerness)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_centers = self.get_centers(featmap_sizes, self.strides,
                                        cls_scores[0].dtype,
                                        cls_scores[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(
                cls_score_list, bbox_pred_list, centerness_pred_list,
                mlvl_centers, img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          mlvl_centers,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_centers)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for cls_score, bbox_pred, centerness, centers in zip(
                cls_scores, bbox_preds, centernesses, mlvl_centers):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.num_classes)
            centerness = centerness.permute(1, 2, 0).reshape(-1)
            centerness = centerness.sigmoid()
            scores = cls_score.sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)

            candidate_mask = scores > cfg.score_thr
            num_candidates = candidate_mask.sum().float()
            scores *= centerness[..., None]

            scores = scores[candidate_mask]
            candidate_mask_nonzero = candidate_mask.nonzero()
            candidate_inds = candidate_mask_nonzero[:, 0]
            labels = candidate_mask_nonzero[:, 1] + 1
            bbox_pred = bbox_pred[candidate_inds]
            centers = centers[candidate_inds]
            if nms_pre > 0 and num_candidates > nms_pre:
                _, topk_inds = scores.topk(nms_pre)
                scores = scores[topk_inds]
                bbox_pred = bbox_pred[topk_inds, :]
                labels = labels[topk_inds]
                centers = centers[topk_inds, :]

            x1 = centers[:, 0] - bbox_pred[:, 0]
            y1 = centers[:, 1] - bbox_pred[:, 1]
            x2 = centers[:, 0] + bbox_pred[:, 2]
            y2 = centers[:, 1] + bbox_pred[:, 3]
            x1 = x1.clamp(min=0, max=img_shape[1] - 1)
            y1 = y1.clamp(min=0, max=img_shape[0] - 1)
            x2 = x2.clamp(min=0, max=img_shape[1] - 1)
            y2 = y2.clamp(min=0, max=img_shape[0] - 1)
            bboxes = torch.stack([x1, y1, x2, y2], -1)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)
        det_bboxes, det_labels = singleclass_nms(mlvl_bboxes, mlvl_scores,
                                                 mlvl_labels, self.num_classes,
                                                 cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels

    def get_centers(self, feat_sizes, strides, dtype, device):
        return multi_apply(
            self.get_centers_single,
            feat_sizes,
            strides,
            dtype=dtype,
            device=device)[0]

    def get_centers_single(self, feat_size, stride, dtype, device):
        h, w = feat_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        centers = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return [centers]
