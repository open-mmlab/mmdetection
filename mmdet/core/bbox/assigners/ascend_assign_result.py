# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.utils import util_mixins


class AscendAssignResult(util_mixins.NiceRepr):
    """Stores ascend assignments between predicted and truth boxes.

    Arguments:
        batch_num_gts (list[int]): the number of truth boxes considered.
        batch_pos_mask (IntTensor): Positive samples mask in all images.
        batch_neg_mask (IntTensor): Negative samples mask in all images.
        batch_max_overlaps (FloatTensor): The max overlaps of all bboxes
            and ground truth boxes.
        batch_anchor_gt_indes(None | LongTensor): The assigned truth
            box index of all anchors.
        batch_anchor_gt_labels(None | LongTensor): The gt labels
            of all anchors
    """

    def __init__(self,
                 batch_num_gts,
                 batch_pos_mask,
                 batch_neg_mask,
                 batch_max_overlaps,
                 batch_anchor_gt_indes=None,
                 batch_anchor_gt_labels=None):
        self.batch_num_gts = batch_num_gts
        self.batch_pos_mask = batch_pos_mask
        self.batch_neg_mask = batch_neg_mask
        self.batch_max_overlaps = batch_max_overlaps
        self.batch_anchor_gt_indes = batch_anchor_gt_indes
        self.batch_anchor_gt_labels = batch_anchor_gt_labels
        # Interface for possible user-defined properties
        self._extra_properties = {}
