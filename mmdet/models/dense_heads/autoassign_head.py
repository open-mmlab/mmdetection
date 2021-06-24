import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply
from mmdet.core.bbox import bbox_overlaps
from mmdet.models import HEADS
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.dense_heads.fcos_head import FCOSHead
from mmdet.models.dense_heads.paa_head import levels_to_images

EPS = 1e-12


class CenterPrior(nn.Module):
    """Center Weighting module to adjust the category-specific prior
    distributions.

    Args:
        force_topk (bool): When no point falls into gt_bbox, forcibly
            select the k points closest to the center to calculate
            the center prior. Defaults to False.
        topk (int): The number of points used to calculate the
            center prior when no point falls in gt_bbox. Only work when
            force_topk if True. Defaults to 9.
        num_classes (int): The class number of dataset. Defaults to 80.
        strides (tuple[int]): The stride of each input feature map. Defaults
            to (8, 16, 32, 64, 128).
    """

    def __init__(self,
                 force_topk=False,
                 topk=9,
                 num_classes=80,
                 strides=(8, 16, 32, 64, 128)):
        super(CenterPrior, self).__init__()
        self.mean = nn.Parameter(torch.zeros(num_classes, 2))
        self.sigma = nn.Parameter(torch.ones(num_classes, 2))
        self.strides = strides
        self.force_topk = force_topk
        self.topk = topk

    def forward(self, anchor_points_list, gt_bboxes, labels,
                inside_gt_bbox_mask):
        """Get the center prior of each point on the feature map for each
        instance.

        Args:
            anchor_points_list (list[Tensor]): list of coordinate
                of points on feature map. Each with shape
                (num_points, 2).
            gt_bboxes (Tensor): The gt_bboxes with shape of
                (num_gt, 4).
            labels (Tensor): The gt_labels with shape of (num_gt).
            inside_gt_bbox_mask (Tensor): Tensor of bool type,
                with shape of (num_points, num_gt), each
                value is used to mark whether this point falls
                within a certain gt.

        Returns:
            tuple(Tensor):

                - center_prior_weights(Tensor): Float tensor with shape \
                    of (num_points, num_gt). Each value represents \
                    the center weighting coefficient.
                - inside_gt_bbox_mask (Tensor): Tensor of bool type, \
                    with shape of (num_points, num_gt), each \
                    value is used to mark whether this point falls \
                    within a certain gt or is the topk nearest points for \
                    a specific gt_bbox.
        """
        inside_gt_bbox_mask = inside_gt_bbox_mask.clone()
        num_gts = len(labels)
        num_points = sum([len(item) for item in anchor_points_list])
        if num_gts == 0:
            return gt_bboxes.new_zeros(num_points,
                                       num_gts), inside_gt_bbox_mask
        center_prior_list = []
        for slvl_points, stride in zip(anchor_points_list, self.strides):
            # slvl_points: points from single level in FPN, has shape (h*w, 2)
            # single_level_points has shape (h*w, num_gt, 2)
            single_level_points = slvl_points[:, None, :].expand(
                (slvl_points.size(0), len(gt_bboxes), 2))
            gt_center_x = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2)
            gt_center_y = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2)
            gt_center = torch.stack((gt_center_x, gt_center_y), dim=1)
            gt_center = gt_center[None]
            # instance_center has shape (1, num_gt, 2)
            instance_center = self.mean[labels][None]
            # instance_sigma has shape (1, num_gt, 2)
            instance_sigma = self.sigma[labels][None]
            # distance has shape (num_points, num_gt, 2)
            distance = (((single_level_points - gt_center) / float(stride) -
                         instance_center)**2)
            center_prior = torch.exp(-distance /
                                     (2 * instance_sigma**2)).prod(dim=-1)
            center_prior_list.append(center_prior)
        center_prior_weights = torch.cat(center_prior_list, dim=0)

        if self.force_topk:
            gt_inds_no_points_inside = torch.nonzero(
                inside_gt_bbox_mask.sum(0) == 0).reshape(-1)
            if gt_inds_no_points_inside.numel():
                topk_center_index = \
                    center_prior_weights[:, gt_inds_no_points_inside].topk(
                                                             self.topk,
                                                             dim=0)[1]
                temp_mask = inside_gt_bbox_mask[:, gt_inds_no_points_inside]
                inside_gt_bbox_mask[:, gt_inds_no_points_inside] = \
                    torch.scatter(temp_mask,
                                  dim=0,
                                  index=topk_center_index,
                                  src=torch.ones_like(
                                    topk_center_index,
                                    dtype=torch.bool))

        center_prior_weights[~inside_gt_bbox_mask] = 0
        return center_prior_weights, inside_gt_bbox_mask


@HEADS.register_module()
class AutoAssignHead(FCOSHead):
    """AutoAssignHead head used in `AutoAssign.

    <https://arxiv.org/abs/2007.03496>`_.

    Args:
        force_topk (bool): Used in center prior initialization to
            handle extremely small gt. Default is False.
        topk (int): The number of points used to calculate the
            center prior when no point falls in gt_bbox. Only work when
            force_topk if True. Defaults to 9.
        pos_loss_weight (float): The loss weight of positive loss
            and with default value 0.25.
        neg_loss_weight (float): The loss weight of negative loss
            and with default value 0.75.
        center_loss_weight (float): The loss weight of center prior
            loss and with default value 0.75.
    """

    def __init__(self,
                 *args,
                 force_topk=False,
                 topk=9,
                 pos_loss_weight=0.25,
                 neg_loss_weight=0.75,
                 center_loss_weight=0.75,
                 **kwargs):
        super().__init__(*args, conv_bias=True, **kwargs)
        self.center_prior = CenterPrior(
            force_topk=force_topk,
            topk=topk,
            num_classes=self.num_classes,
            strides=self.strides)
        self.pos_loss_weight = pos_loss_weight
        self.neg_loss_weight = neg_loss_weight
        self.center_loss_weight = center_loss_weight

    def init_weights(self):
        """Initialize weights of the head.

        In particular, we have special initialization for classified conv's and
        regression conv's bias
        """

        super(AutoAssignHead, self).init_weights()
        bias_cls = bias_init_with_prob(0.02)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01, bias=4.0)

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Almost the same as the implementation in fcos, we remove half stride
        offset to align with the original implementation."""

        y, x = super(FCOSHead,
                     self)._get_points_single(featmap_size, stride, dtype,
                                              device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1)
        return points

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super(
            FCOSHead, self).forward_single(x)
        centerness = self.conv_centerness(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        bbox_pred = F.relu(bbox_pred)
        bbox_pred *= stride
        return cls_score, bbox_pred, centerness

    def get_pos_loss_single(self, cls_score, objectness, reg_loss, gt_labels,
                            center_prior_weights):
        """Calculate the positive loss of all points in gt_bboxes.

        Args:
            cls_score (Tensor): All category scores for each point on
                the feature map. The shape is (num_points, num_class).
            objectness (Tensor): Foreground probability of all points,
                has shape (num_points, 1).
            reg_loss (Tensor): The regression loss of each gt_bbox and each
                prediction box, has shape of (num_points, num_gt).
            gt_labels (Tensor): The zeros based gt_labels of all gt
                with shape of (num_gt,).
            center_prior_weights (Tensor): Float tensor with shape
                of (num_points, num_gt). Each value represents
                the center weighting coefficient.

        Returns:
            tuple[Tensor]:

                - pos_loss (Tensor): The positive loss of all points
                  in the gt_bboxes.
        """
        # p_loc: localization confidence
        p_loc = torch.exp(-reg_loss)
        # p_cls: classification confidence
        p_cls = (cls_score * objectness)[:, gt_labels]
        # p_pos: joint confidence indicator
        p_pos = p_cls * p_loc

        # 3 is a hyper-parameter to control the contributions of high and
        # low confidence locations towards positive losses.
        confidence_weight = torch.exp(p_pos * 3)
        p_pos_weight = (confidence_weight * center_prior_weights) / (
            (confidence_weight * center_prior_weights).sum(
                0, keepdim=True)).clamp(min=EPS)
        reweighted_p_pos = (p_pos * p_pos_weight).sum(0)
        pos_loss = F.binary_cross_entropy(
            reweighted_p_pos,
            torch.ones_like(reweighted_p_pos),
            reduction='none')
        pos_loss = pos_loss.sum() * self.pos_loss_weight
        return pos_loss,

    def get_neg_loss_single(self, cls_score, objectness, gt_labels, ious,
                            inside_gt_bbox_mask):
        """Calculate the negative loss of all points in feature map.

        Args:
            cls_score (Tensor): All category scores for each point on
                the feature map. The shape is (num_points, num_class).
            objectness (Tensor): Foreground probability of all points
                and is shape of (num_points, 1).
            gt_labels (Tensor): The zeros based label of all gt with shape of
                (num_gt).
            ious (Tensor): Float tensor with shape of (num_points, num_gt).
                Each value represent the iou of pred_bbox and gt_bboxes.
            inside_gt_bbox_mask (Tensor): Tensor of bool type,
                with shape of (num_points, num_gt), each
                value is used to mark whether this point falls
                within a certain gt.

        Returns:
            tuple[Tensor]:

                - neg_loss (Tensor): The negative loss of all points
                  in the feature map.
        """
        num_gts = len(gt_labels)
        joint_conf = (cls_score * objectness)
        p_neg_weight = torch.ones_like(joint_conf)
        if num_gts > 0:
            # the order of dinmension would affect the value of
            # p_neg_weight, we strictly follow the original
            # implementation.
            inside_gt_bbox_mask = inside_gt_bbox_mask.permute(1, 0)
            ious = ious.permute(1, 0)

            foreground_idxs = torch.nonzero(inside_gt_bbox_mask, as_tuple=True)
            temp_weight = (1 / (1 - ious[foreground_idxs]).clamp_(EPS))

            def normalize(x):
                return (x - x.min() + EPS) / (x.max() - x.min() + EPS)

            for instance_idx in range(num_gts):
                idxs = foreground_idxs[0] == instance_idx
                if idxs.any():
                    temp_weight[idxs] = normalize(temp_weight[idxs])

            p_neg_weight[foreground_idxs[1],
                         gt_labels[foreground_idxs[0]]] = 1 - temp_weight

        logits = (joint_conf * p_neg_weight)
        neg_loss = (
            logits**2 * F.binary_cross_entropy(
                logits, torch.zeros_like(logits), reduction='none'))
        neg_loss = neg_loss.sum() * self.neg_loss_weight
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
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            objectnesses (list[Tensor]): objectness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        all_num_gt = sum([len(item) for item in gt_bboxes])
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        inside_gt_bbox_mask_list, bbox_targets_list = self.get_targets(
            all_level_points, gt_bboxes)

        center_prior_weight_list = []
        temp_inside_gt_bbox_mask_list = []
        for gt_bboxe, gt_label, inside_gt_bbox_mask in zip(
                gt_bboxes, gt_labels, inside_gt_bbox_mask_list):
            center_prior_weight, inside_gt_bbox_mask = \
                self.center_prior(all_level_points, gt_bboxe, gt_label,
                                  inside_gt_bbox_mask)
            center_prior_weight_list.append(center_prior_weight)
            temp_inside_gt_bbox_mask_list.append(inside_gt_bbox_mask)
        inside_gt_bbox_mask_list = temp_inside_gt_bbox_mask_list

        mlvl_points = torch.cat(all_level_points, dim=0)
        bbox_preds = levels_to_images(bbox_preds)
        cls_scores = levels_to_images(cls_scores)
        objectnesses = levels_to_images(objectnesses)

        reg_loss_list = []
        ious_list = []
        num_points = len(mlvl_points)

        for bbox_pred, gt_bboxe, inside_gt_bbox_mask in zip(
                bbox_preds, bbox_targets_list, inside_gt_bbox_mask_list):
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
                ious = ious.reshape(num_points, temp_num_gt)
                if temp_num_gt:
                    ious = ious.max(
                        dim=-1, keepdim=True).values.repeat(1, temp_num_gt)
                else:
                    ious = ious.new_zeros(num_points, temp_num_gt)
                ious[~inside_gt_bbox_mask] = 0
                ious_list.append(ious)
            loss_bbox = self.loss_bbox(
                decoded_bbox_preds,
                decoded_target_preds,
                weight=None,
                reduction_override='none')
            reg_loss_list.append(loss_bbox.reshape(num_points, temp_num_gt))

        cls_scores = [item.sigmoid() for item in cls_scores]
        objectnesses = [item.sigmoid() for item in objectnesses]
        pos_loss_list, = multi_apply(self.get_pos_loss_single, cls_scores,
                                     objectnesses, reg_loss_list, gt_labels,
                                     center_prior_weight_list)
        pos_avg_factor = reduce_mean(
            bbox_pred.new_tensor(all_num_gt)).clamp_(min=1)
        pos_loss = sum(pos_loss_list) / pos_avg_factor

        neg_loss_list, = multi_apply(self.get_neg_loss_single, cls_scores,
                                     objectnesses, gt_labels, ious_list,
                                     inside_gt_bbox_mask_list)
        neg_avg_factor = sum(item.data.sum()
                             for item in center_prior_weight_list)
        neg_avg_factor = reduce_mean(neg_avg_factor).clamp_(min=1)
        neg_loss = sum(neg_loss_list) / neg_avg_factor

        center_loss = []
        for i in range(len(img_metas)):

            if inside_gt_bbox_mask_list[i].any():
                center_loss.append(
                    len(gt_bboxes[i]) /
                    center_prior_weight_list[i].sum().clamp_(min=EPS))
            # when width or height of gt_bbox is smaller than stride of p3
            else:
                center_loss.append(center_prior_weight_list[i].sum() * 0)

        center_loss = torch.stack(center_loss).mean() * self.center_loss_weight

        # avoid dead lock in DDP
        if all_num_gt == 0:
            pos_loss = bbox_preds[0].sum() * 0
            dummy_center_prior_loss = self.center_prior.mean.sum(
            ) * 0 + self.center_prior.sigma.sum() * 0
            center_loss = objectnesses[0].sum() * 0 + dummy_center_prior_loss

        loss = dict(
            loss_pos=pos_loss, loss_neg=neg_loss, loss_center=center_loss)

        return loss

    def get_targets(self, points, gt_bboxes_list):
        """Compute regression targets and each point inside or outside gt_bbox
        in multiple images.

        Args:
            points (list[Tensor]): Points of all fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).

        Returns:
            tuple(list[Tensor]):

                - inside_gt_bbox_mask_list (list[Tensor]): Each
                  Tensor is with bool type and shape of
                  (num_points, num_gt), each value
                  is used to mark whether this point falls
                  within a certain gt.
                - concat_lvl_bbox_targets (list[Tensor]): BBox
                  targets of each level. Each tensor has shape
                  (num_points, num_gt, 4).
        """

        concat_points = torch.cat(points, dim=0)
        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        inside_gt_bbox_mask_list, bbox_targets_list = multi_apply(
            self._get_target_single, gt_bboxes_list, points=concat_points)
        bbox_targets_list = [
            list(bbox_targets.split(num_points, 0))
            for bbox_targets in bbox_targets_list
        ]
        concat_lvl_bbox_targets = [
            torch.cat(item, dim=0) for item in bbox_targets_list
        ]
        return inside_gt_bbox_mask_list, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, points):
        """Compute regression targets and each point inside or outside gt_bbox
        for a single image.

        Args:
            gt_bboxes (Tensor): gt_bbox of single image, has shape
                (num_gt, 4).
            points (Tensor): Points of all fpn level, has shape
                (num_points, 2).

        Returns:
            tuple[Tensor]: Containing the following Tensors:

                - inside_gt_bbox_mask (Tensor): Bool tensor with shape
                  (num_points, num_gt), each value is used to mark
                  whether this point falls within a certain gt.
                - bbox_targets (Tensor): BBox targets of each points with
                  each gt_bboxes, has shape (num_points, num_gt, 4).
        """
        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None]
        ys = ys[:, None]
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        if num_gts:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.new_zeros((num_points, num_gts),
                                                         dtype=torch.bool)

        return inside_gt_bbox_mask, bbox_targets
