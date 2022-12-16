# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.utils import util_mixins


class AscendAssignResult(util_mixins.NiceRepr):
    """Stores assignments between predicted and truth boxes.

    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment

        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.

        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.

        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.

    Example:
        >>> # An assign result between 4 predicted boxes and 9 true boxes
        >>> # where only two boxes were assigned.
        >>> num_gts = 9
        >>> max_overlaps = torch.LongTensor([0, .5, .9, 0])
        >>> gt_inds = torch.LongTensor([-1, 1, 2, 0])
        >>> labels = torch.LongTensor([0, 3, 4, 0])
        >>> self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(4,), max_overlaps.shape=(4,),
                      labels.shape=(4,))>
        >>> # Force addition of gt labels (when adding gt as proposals)
        >>> new_labels = torch.LongTensor([3, 4, 5])
        >>> self.add_gt_(new_labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(7,), max_overlaps.shape=(7,),
                      labels.shape=(7,))>
    """

    def __init__(self, concat_num_gts, concat_pos_mask, concat_neg_mask,
                 concat_max_overlaps, concat_anchor_gt_indes=None, concat_anchor_gt_labels=None):
        self.concat_num_gts = concat_num_gts
        self.concat_pos_mask = concat_pos_mask
        self.concat_neg_mask = concat_neg_mask
        self.concat_max_overlaps = concat_max_overlaps
        self.concat_anchor_gt_indes = concat_anchor_gt_indes
        self.concat_anchor_gt_labels = concat_anchor_gt_labels
        # Interface for possible user-defined properties
        self._extra_properties = {}
