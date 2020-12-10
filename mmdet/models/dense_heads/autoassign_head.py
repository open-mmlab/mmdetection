import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, bias_init_with_prob
from mmdet.core import distance2bbox, force_fp32, multi_apply
from mmdet.core.bbox import bbox_overlaps
from mmdet.models import HEADS
from mmdet.models.dense_heads.fcos_head import FCOSHead
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.dense_heads.paa_head import levels_to_images

EPS = 1e-12

class CenterPrior(nn.Module):

    def __init__(self,
                 type='category',
                 force_inside=True,
                 class_num=80,
                 stride=(8, 16, 32, 64, 128)):
        assert type in ('fixed', 'shared', 'category')
        # TODO: support fixed and shared
        super(CenterPrior, self).__init__()
        self.mean = nn.Parameter(torch.zeros(class_num, 2).float())
        self.sigma = nn.Parameter(torch.ones(class_num,2).float() )
        self.stride = stride
        self.force_inside = force_inside

    def forward(self, anchor_points_list, gt_bboxes, labels,
                inside_gt_bbox_mask):
        center_prior_list = []
        for slvl_points, stride in zip(anchor_points_list, self.stride):
            single_level_points = slvl_points[:, None, :].expand(
                (slvl_points.size(0), len(gt_bboxes), 2))
            gt_center_x = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2)
            gt_center_y = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2)
            gt_center = torch.stack((gt_center_x, gt_center_y), dim=1)
            gt_center = gt_center[None, :, :]
            # shape (1, num_gt, 2)
            cat_prior_center = self.mean[labels][None, :]
            # shape (1, num_gt,2)
            cat_prior_sigma = self.sigma[labels][None, :]
            distance = (((single_level_points - gt_center) / float(stride) -
                         cat_prior_center)**2)
            center_prior = torch.exp(-distance /
                                     (2 * cat_prior_sigma**2).clamp(min=EPS)).prod(dim=-1)
            center_prior_list.append(center_prior)
        center_prior_weights = torch.cat(center_prior_list, dim=0)
        if self.force_inside:
            # if no any points in gt, we select top 9 as center_prior
            no_inside_gt_index = torch.nonzero(
                inside_gt_bbox_mask.sum(0) == 0).reshape(-1)
            topk_center_index = center_prior[:, no_inside_gt_index].topk(
                9, dim=0)[1]
            inside_gt_bbox_mask.scatter_(
                dim=0,
                index=topk_center_index,
                src=torch.ones_like(topk_center_index, dtype=torch.bool))
        center_prior_weights[~inside_gt_bbox_mask] = 0
        return center_prior_weights


@HEADS.register_module()
class AutoAssignHead(FCOSHead):

    def __init__(self, *args, force_inside=True,centerness_on_reg=True, **kwargs):
        super().__init__(*args,conv_bias=True,centerness_on_reg=centerness_on_reg ,**kwargs)
        self.center_prior = CenterPrior(force_inside=force_inside)
        self.force_inside = force_inside

    def init_weights(self):
        """Initialize weights of the head."""
        super(AutoAssignHead, self).init_weights()
        bias_cls = bias_init_with_prob(0.02)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01, bias=4.0)


    def get_pos_loss_single(self, cls_score, objectness, reg_loss, label,
                     center_prior_weights):
        # TODO: only sigmoid once with neg loss
        cls_score = cls_score.sigmoid()
        objectness = objectness.sigmoid()
        p_pos = torch.exp(-5 * reg_loss) * (cls_score * objectness)[:, label]
        p_pos_weight = torch.exp(p_pos * 3)
        p_pos_weight = (p_pos_weight * center_prior_weights) / (
            (p_pos_weight * center_prior_weights).sum(
                0, keepdim=True)).clamp(min=EPS)
        reweight_pos = (p_pos * p_pos_weight).sum(0)
        pos_loss = F.binary_cross_entropy(
            reweight_pos, torch.ones_like(reweight_pos), reduction='none')
        return pos_loss.sum(),

    def get_neg_weight(self, gt_labels, neg_weight):
        # TODO to speed up with tensor operation as FreeAnchor

        temp_label = gt_labels.clone()
        label_list = []
        neg_weight_list = []
        while not (temp_label == self.num_classes).all():
            inds = (~(temp_label == self.num_classes)).nonzero().reshape(-1)
            label_mask = temp_label == temp_label[inds[0]]
            temp_label[label_mask] = self.num_classes
            temp_weight = neg_weight[:, label_mask].min(-1)[0]
            label_list.append(gt_labels[inds[0]])
            neg_weight_list.append(temp_weight)
        group_neg_weight = torch.stack(neg_weight_list, dim=1)
        group_label = torch.stack(label_list)
        return group_neg_weight, group_label

    def get_neg_loss_single(self, cls_score, objectness, gt_labels, ious,):

        cls_score = cls_score.sigmoid()
        objectness = objectness.sigmoid()
        pred_cls = (cls_score * objectness)
        pred_weight = torch.ones_like(cls_score)
        temp_weight = (1 / (1 - ious))
        # TODO change to gt dim first
        iou_fuc_min = temp_weight.min(0)[0][None, :]
        iou_fuc_max = temp_weight.max(0)[0][None, :]
        neg_weight = 1 - (temp_weight - iou_fuc_min) / (
            iou_fuc_max - iou_fuc_min).clamp(min=EPS)
        neg_weight, gt_labels = self.get_neg_weight(gt_labels, neg_weight)
        scatter_inds = gt_labels[None, :].expand_as(neg_weight)
        pred_weight.scatter_(dim=1, index=scatter_inds, src=neg_weight)
        pred_cls = (pred_cls * pred_weight).clamp(min=EPS)
        pred_cls = -(((1 / (pred_cls)) - 1).log())
        neg_loss = self.loss_cls(
            pred_cls,
            neg_weight.new_zeros((len(pred_cls)),
                                 dtype=torch.long).fill_(self.num_classes),
            None,
            avg_factor=1)
        return neg_loss,

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        all_num_gt = max(sum([len(item) for item in gt_bboxes]), 1)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        inside_gt_bbox_mask_list, bbox_targets_list = self.get_targets(
            all_level_points, gt_bboxes, gt_labels)

        center_prior_weight_list = []
        for gt_bboxe, gt_label, inside_gt_bbox_mask in zip(
                gt_bboxes, gt_labels, inside_gt_bbox_mask_list):
            center_prior_weight_list.append(
                self.center_prior(all_level_points, gt_bboxe, gt_label,
                                  inside_gt_bbox_mask))

        mlvl_points = torch.cat(all_level_points, dim=0)

        bbox_preds = levels_to_images(bbox_preds)
        cls_scores = levels_to_images(cls_scores)
        objectnesses = levels_to_images(objectnesses)

        reg_loss_list = []
        ious_list = []
        num_points = len(mlvl_points)

        for bbox_pred, gt_bboxe in zip(bbox_preds, bbox_targets_list):
            temp_num_gt = gt_bboxe.size(1)
            expand_mlvl_points = mlvl_points[:, None, :].expand(
                num_points, temp_num_gt, 2).reshape(-1, 2)
            gt_bboxe = gt_bboxe.reshape(-1, 4)
            expand_bbox_pred = bbox_pred[:, None, :].expand(
                num_points, temp_num_gt, 4).reshape(-1, 4)
            decoded_bbox_preds = distance2bbox(expand_mlvl_points,
                                               expand_bbox_pred)
            decoded_target_preds = distance2bbox(expand_mlvl_points, gt_bboxe)
            with torch.no_grad():
                ious = bbox_overlaps(
                    decoded_bbox_preds, decoded_target_preds, is_aligned=True)
                assert torch.isnan(ious).sum() == 0
                ious_list.append(ious.reshape(num_points, temp_num_gt))
            loss_bbox = self.loss_bbox(
                decoded_bbox_preds,
                decoded_target_preds,
                weight=None,
                reduction_override='none')
            nan_mask = torch.isnan(loss_bbox)
            loss_bbox[nan_mask] = 2
            reg_loss_list.append(loss_bbox.reshape(num_points, temp_num_gt))

        pos_loss_list, = multi_apply(
            self.get_pos_loss_single,
            cls_scores,
            objectnesses,
            reg_loss_list,
            gt_labels,
            center_prior_weight_list
        )
        pos_norm = reduce_mean(bbox_pred.new_tensor(all_num_gt))
        pos_loss = sum(pos_loss_list) / pos_norm
        neg_loss_list, = multi_apply(
            self.get_neg_loss_single,
            cls_scores,
            objectnesses,
            gt_labels,
            ious_list,
        )

        neg_norm = sum(item.data.sum() for item in center_prior_weight_list)
        neg_norm = reduce_mean(neg_norm)
        neg_loss = sum(neg_loss_list) / max(neg_norm, 1)

        center_loss = []
        for i in range(len(img_metas)):
            center_loss.append(
                len(gt_bboxes[i]) /
                center_prior_weight_list[i].sum().clamp_(min=EPS))
        center_loss = torch.stack(center_loss).mean()

        if all_num_gt == 0:
            pos_loss = bbox_preds[0].sum() * 0
            center_loss = objectnesses[0].sum()* 0

        return dict(
            loss_pos=pos_loss * 0.25,
            loss_neg=neg_loss,
            loss_center=center_loss * 0.75)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):

        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        inside_gt_bbox_mask_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        bbox_targets_list = [
            list(bbox_targets.split(num_points, 0))
            for bbox_targets in bbox_targets_list
        ]
        if self.norm_on_bbox:
            for i in range(num_levels):
                for j in range(len(bbox_targets_list)):
                    bbox_targets_list[j][
                        i] = bbox_targets_list[j][i] / self.strides[i]
        concat_lvl_bbox_targets = [
            torch.cat(item, dim=0) for item in bbox_targets_list
        ]
        return inside_gt_bbox_mask_list, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):

        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        # TODO: check when no any gt in imgs
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        return inside_gt_bbox_mask, bbox_targets
