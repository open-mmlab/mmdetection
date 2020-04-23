import numpy as np
import torch
from mmcv.cnn import normal_init

from mmdet.core import (anchor_inside_flags, force_fp32, images_to_levels,
                        multi_apply, unmap)

from ..losses.utils import weight_reduce_loss
from ..registry import HEADS
from .retina_head import RetinaHead


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
                 conv_cfg=None,
                 norm_cfg=None,
                 effective_threshold=0.2,
                 ignore_threshold=0.2,
                 target_normalizer=1.0,
                 **kwargs):
        self.effective_threshold = effective_threshold
        self.ignore_threshold = ignore_threshold
        self.target_normalizer = target_normalizer
        super(FSAFHead, self).__init__(num_classes, in_channels, stacked_convs,
                                       conv_cfg, norm_cfg, **kwargs)

    def forward_single(self, x):
        cls_score, bbox_pred = super(FSAFHead, self).forward_single(x)
        return cls_score, self.relu(bbox_pred)
        # TBLR encoder only accepts positive bbox_pred

    def init_weights(self):
        super(FSAFHead, self).init_weights()
        normal_init(self.retina_reg, std=0.01, bias=0.25)
        # the positive bias in self.retina_reg conv is to prevent predicted \
        #  bbox with 0 area

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in
            a single image.

            Most of the codes are the same with the base class
              ::obj::`AnchorHead`, except that it also collects and returns
              the matched gt index in the image (from 0 to num_gt-1). If the
              pixel is not matched to any gt, the corresponding value in
              pos_gt_inds is -1.
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 6
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags.type(torch.bool), :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.background_label,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_gt_inds = anchors.new_full((num_valid_anchors,),
                                       -1,
                                       dtype=torch.long)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1. / 4.  # avg in tblr dims
            pos_gt_inds[pos_inds] = sampling_result.pos_assigned_gt_inds
            if gt_labels is None:
                # only rpn gives gt_labels as None, this time FG is 1
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            pos_gt_inds = unmap(
                pos_gt_inds, num_total_anchors, inside_flags, fill=-1)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, pos_gt_inds)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generator.base_anchors)
        batch_size = len(gt_bboxes)
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg,
         pos_assigned_gt_inds_list) = cls_reg_targets

        num_gts = np.array(list(map(len, gt_labels)))
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        for i in range(len(bbox_preds)):
            bbox_preds[i] = bbox_preds[i].clamp(min=1e-4)
            # avoid 0 area of the predicted bbox
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        cum_num_gts = list(np.cumsum(num_gts))
        for i, assign in enumerate(pos_assigned_gt_inds_list):
            for j in range(1, batch_size):
                assign[j][assign[j] >= 0] += int(cum_num_gts[j - 1])
            pos_assigned_gt_inds_list[i] = assign.flatten()
            labels_list[i] = labels_list[i].flatten()
        num_gts = sum(map(len, gt_labels))
        with torch.no_grad():
            loss_levels, = multi_apply(
                self.collect_loss_level_single,
                losses_cls,
                losses_bbox,
                pos_assigned_gt_inds_list,
                labels_seq=torch.arange(num_gts, device=device))
            loss_levels = torch.stack(loss_levels, dim=0)
            loss, argmin = loss_levels.min(dim=0)
        losses_cls, losses_bbox, pos_inds = multi_apply(
            self.reassign_loss_single,
            losses_cls,
            losses_bbox,
            pos_assigned_gt_inds_list,
            labels_list,
            list(range(len(losses_cls))),
            min_levels=argmin)

        num_pos = torch.cat(pos_inds, 0).sum().float().clamp(min=1e-3)
        # clamp to 1e-3 to prevent 0/0
        acc = self.calculate_accuracy(cls_scores, labels_list, pos_inds)
        for i in range(len(losses_cls)):
            losses_cls[i] /= num_pos
            losses_bbox[i] /= num_pos
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            num_pos=num_pos / batch_size,
            accuracy=acc)

    def calculate_accuracy(self, cls_scores, labels_list, pos_inds):
        with torch.no_grad():
            num_pos = torch.cat(pos_inds, 0).sum().float().clamp(min=1e-3)
            num_class = cls_scores[0].size(1)
            scores = [
                cls.permute(0, 2, 3, 1).reshape(-1, num_class)[pos]
                for cls, pos in zip(cls_scores, pos_inds)
            ]
            labels = [
                l.reshape(-1)[pos] for l, pos in zip(labels_list, pos_inds)
            ]

            def argmax(x):
                return x.argmax(1) if x.numel() > 0 else -100

            num_correct = sum([(argmax(score) == label).sum()
                               for score, label in zip(scores, labels)])
            return num_correct.float() / num_pos

    def collect_loss_level_single(self, cls_loss, reg_loss,
                                  assigned_gt_inds, labels_seq):
        """Get the average loss in each FPN level w.r.t. each gt label

        Args:
            cls_loss (tensor): classification loss of each feature map pixel,
              shape (num_anchor, num_class)
            reg_loss (tensor): regression loss of each feature map pixel,
              shape (num_anchor, 4)
            assigned_gt_inds (tensor): shape (num_anchor), indicating
              which gt the prior is assigned to (-1: no assignment)
            labels_seq: The rank of labels. shape (num_gt)

        Returns:
            shape: (num_gt), average loss of each gt in this level
        """
        if len(reg_loss.shape) == 2:  # iou loss has shape [num_prior, 4]
            reg_loss = reg_loss.sum(dim=-1)  # sum loss in tblr dims
        # total loss at each feature map point
        if len(cls_loss.shape) == 2:
            cls_loss = cls_loss.sum(dim=-1)  # sum loss in class dims
        loss = cls_loss + reg_loss
        assert loss.size(0) == assigned_gt_inds.size(0)
        # default loss value is 1e6 for a layer where no anchor is positive
        losses_ = loss.new_full(labels_seq.shape, 1e6)
        for i, l in enumerate(labels_seq):
            match = assigned_gt_inds == l
            if match.any():
                losses_[i] = loss[match].mean()
        return losses_,

    def reassign_loss_single(self, cls_loss, reg_loss, assigned_gt_inds,
                             labels, level, min_levels):
        """Reassign loss values at each level.

         Reassign loss values at each level by masking those where the
          pre-calculated loss is too large

        Args:
            cls_loss (tensor): shape (num_anchors, num_classes)
              classification loss
            reg_loss (tensor): shape (num_anchors) regression loss
            assigned_gt_inds (tensor): shape (num_anchors),
              the gt indices that each positive anchor corresponds to.
              (-1 if it is a negative one)
            labels (tensor): shape (num_anchors). Label assigned to each pixel
            level (int): the current level index in the
              pyramid (0-4 for RetinaNet)
            min_levels (tensor): shape (num_gts),
              the best-matching level for each gt

        Returns:
            cls_loss: shape (num_anchors, num_classes).
              Corrected classification loss
            reg_loss: shape (num_anchors). Corrected regression loss
            pos_flags: shape (num_anchors). Indicating final postive anchors
        """
        loc_weight = torch.ones_like(reg_loss)
        cls_weight = torch.ones_like(cls_loss)
        pos_flags = assigned_gt_inds >= 0  # positive pixel flag
        pos_indices = torch.nonzero(pos_flags).flatten()

        if pos_flags.any():  # pos pixels exist
            pos_assigned_gt_inds = assigned_gt_inds[pos_flags]
            zeroing_indices = (min_levels[pos_assigned_gt_inds] != level)
            neg_indices = pos_indices[zeroing_indices]

            if neg_indices.numel():
                pos_flags[neg_indices] = 0
                loc_weight[neg_indices] = 0
                # Only the weight corresponding to the label is
                #  zeroed out if not selected
                zeroing_labels = labels[neg_indices]
                # the original labels assigned to the anchor box
                assert (zeroing_labels >= 0).all()
                cls_weight[neg_indices, zeroing_labels] = 0

        # weighted loss for both cls and reg loss
        cls_loss = weight_reduce_loss(cls_loss, cls_weight, reduction='sum')
        reg_loss = weight_reduce_loss(reg_loss, loc_weight, reduction='sum')

        return cls_loss, reg_loss, pos_flags
