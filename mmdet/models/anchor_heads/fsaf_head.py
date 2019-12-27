import numpy as np
import torch

from .retina_head import RetinaHead
from ..losses.utils import weighted_loss, weight_reduce_loss
from ..losses import IoULoss
from ..registry import HEADS, LOSSES
from mmdet.core import (tblr2bboxes, multiclass_nms)
from mmdet.core import (anchor_tblr_target, force_fp32, multi_apply)



@weighted_loss
def iou_loss_tblr(pred, target, eps=1e-6):
    """
    calculate the iou loss when both the prediction and targets are encoded in TBLR format
    Args:
        pred: shape (num_anchor, 4)
        target: shape (num_anchor, 4)
        eps: the minimum iou threshold

    Returns:
        loss: shape (num_anchor), IoU loss

    """
    xt, xb, xl, xr = torch.split(pred, 1, dim=-1)

    # the ground truth position
    gt, gb, gl, gr = torch.split(target, 1, dim=-1)

    # compute the bounding box size
    X = (xt + xb) * (xl + xr)  # AreaX
    G = (gt + gb) * (gl + gr)  # AreaG

    # compute the IOU
    Ih = torch.min(xt, gt) + torch.min(xb, gb)
    Iw = torch.min(xl, gl) + torch.min(xr, gr)

    I = Ih * Iw
    U = (X + G - I).clamp(min=1) # minimum area should be 1

    IoU = I / U
    IoU = IoU.squeeze()
    ious = IoU.clamp(min=eps)
    loss = -ious.log()
    return loss


@LOSSES.register_module
class IoULossTBLR(IoULoss):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(IoULossTBLR, self).__init__(eps, reduction, loss_weight)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        weight = weight.sum(dim=-1) / 4. # iou loss is a scalar!
        loss = self.loss_weight * iou_loss_tblr(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@HEADS.register_module
class FSAFHead(RetinaHead):
    """
    FSAF anchor-free head used in [1].

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors (num_anchors is 1
    for anchor-free methods)

    References:
        .. [1]  https://arxiv.org/pdf/1903.00621.pdf

    Example:
        >>> import torch
        >>> self = FSAFHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes - 1)
        >>> assert box_per_anchor == 4
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 effective_threshold=0.2,
                 ignore_threshold=0.2,
                 target_normalizer=1.0,
                 **kwargs):
        self.effective_threshold = effective_threshold
        self.ignore_threshold = ignore_threshold
        self.target_normalizer = target_normalizer
        super(FSAFHead, self).__init__(num_classes,
                                       in_channels,
                                       stacked_convs,
                                       octave_base_scale,
                                       scales_per_octave,
                                       conv_cfg,
                                       norm_cfg,
                                       **kwargs)

    def forward_single(self, x):
        cls_score, bbox_pred = super(FSAFHead, self).forward_single(x)
        return cls_score, self.relu(bbox_pred)
        # TBLR encoder only accepts positive bbox_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None,
             ):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        batch_size = len(gt_bboxes)
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        cls_reg_targets = anchor_tblr_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_normalizer,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, pos_assigned_gt_inds_list) = cls_reg_targets

        num_gts = np.array(list(map(len, gt_labels)))
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        cum_num_gts = list(np.cumsum(num_gts))
        for i, assign in enumerate(pos_assigned_gt_inds_list):
            for j in range(1, batch_size):
                assign[j][assign[j] >= 0] += int(cum_num_gts[j - 1])
            pos_assigned_gt_inds_list[i] = assign.flatten()
            labels_list[i] = labels_list[i].flatten()
        num_gts = sum(map(len, gt_labels))
        with torch.no_grad():
            loss_levels, = multi_apply(self.collect_loss_level_single,
                                       losses_cls, losses_bbox, pos_assigned_gt_inds_list,
                                       labels_seq=torch.arange(num_gts, device=device))
            loss_levels = torch.stack(loss_levels, dim=0)
            loss, argmin = loss_levels.min(dim=0)
        losses_cls, losses_bbox, pos_inds = multi_apply(self.reassign_loss_single,
                                                        losses_cls,
                                                        losses_bbox,
                                                        pos_assigned_gt_inds_list,
                                                        labels_list,
                                                        list(range(len(losses_cls))),
                                                        min_levels=argmin)
        num_pos = torch.cat(pos_inds, 0).sum().float()
        acc = self.calculate_accuracy(cls_scores, labels_list, pos_inds)
        for i in range(len(losses_cls)):
            losses_cls[i] /= num_pos
            losses_bbox[i] /= num_pos
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, num_pos=num_pos / batch_size, accuracy=acc)

    def calculate_accuracy(self, cls_scores, labels_list, pos_inds):
        with torch.no_grad():
            num_pos = torch.cat(pos_inds, 0).sum().float()
            num_class = cls_scores[0].size(1)
            scores = [cls.permute(0, 2, 3, 1).reshape(-1, num_class)[pos]
             for cls, pos in zip(cls_scores, pos_inds)]
            labels = [l.reshape(-1)[pos] for l, pos in zip(labels_list, pos_inds)]
            argmax = lambda x: x.argmax(1) if x.numel()>0 else -100
            num_correct = sum([(argmax(score) + 1 == label).sum() for score, label in zip(scores, labels)])
            return num_correct.float() / (num_pos+1e-3)

    def collect_loss_level_single(self, cls_loss, reg_loss, pos_assigned_gt_inds, labels_seq):
        """
        Get the average loss in each FPN level w.r.t. each gt label
        Args:
            cls_loss (tensor): classification loss of each feature map pixel, shape (num_anchor, num_class)
            reg_loss (tensor): regression loss of each feature map pixel, shape (num_anchor)
            pos_assigned_gt_inds (tensor): shape (num_anchor), indicating which gt the prior is assigned to
                (-1: no assignment)
            labels_seq: The rank of labels

        Returns:

        """
        loss = cls_loss.sum(dim=-1) + reg_loss # total loss at each feature map point
        match = pos_assigned_gt_inds.reshape(-1).unsqueeze(1) == labels_seq.unsqueeze(0)
        loss_ceiling = loss.new_zeros(1).squeeze() + 1e6
        # default loss value for a layer where no anchor is positive
        losses_ = torch.stack(
            [torch.mean(loss[match[:, i]]) if match[:, i].sum() > 0 else loss_ceiling for i in labels_seq])
        return losses_,

    def reassign_loss_single(self, cls_loss, reg_loss, pos_assigned_gt_inds, labels, level, min_levels):
        """
        Reassign loss values at each level by masking those where the pre-calculated loss is too large
        Args:
            cls_loss (tensor): shape (num_anchors, num_classes) classification loss
            reg_loss (tensor): shape (num_anchors) regression loss
            pos_assigned_gt_inds (tensor): shape (num_anchors),
                the gt indices that each positive anchor corresponds to. (-1 if it is a negative one)
            labels (tensor): shape (num_anchors). Label assigned to each pixel
            level (int): the current level index in the pyramid (0-4 for RetinaNet)
            min_levels (tensor): shape (num_gts), the best-matching level for each gt

        Returns:
            cls_loss: shape (num_anchors, num_classes). Corrected classification loss
            reg_loss: shape (num_anchors). Corrected regression loss
            keep_indices: shape (num_anchors). Indicating final postive anchors
        """

        unmatch_gt_inds = torch.nonzero(min_levels != level)
        # gts indices that unmatch with the current level
        match_gt_inds = torch.nonzero(min_levels == level)
        loc_weight = cls_loss.new_ones(cls_loss.size(0))
        cls_weight = cls_loss.new_ones(cls_loss.size(0), cls_loss.size(1))
        zeroing_indices = (pos_assigned_gt_inds.view(-1, 1) ==
                           unmatch_gt_inds.view(1, -1)).any(dim=-1)
        keep_indices = (pos_assigned_gt_inds.view(-1, 1) ==
                        match_gt_inds.view(1, -1)).any(dim=-1)
        loc_weight[zeroing_indices] = 0

        #only the weight corresponding to the label is zeroed out if not selected
        zeroing_labels = labels[zeroing_indices] - 1 # the original labels assigned to the anchor box
        assert (zeroing_labels>=0).all()
        cls_weight[zeroing_indices, zeroing_labels] = 0

        #weighted loss for both cls and reg loss
        cls_loss = weight_reduce_loss(cls_loss, cls_weight, reduction='sum')
        reg_loss = weight_reduce_loss(reg_loss, loc_weight, reduction='sum')
        return cls_loss, reg_loss, keep_indices

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = tblr2bboxes(anchors, bbox_pred,
                                self.target_normalizer, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels