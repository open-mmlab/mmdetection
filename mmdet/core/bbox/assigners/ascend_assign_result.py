# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.utils import util_mixins


class AscendAssignResult(util_mixins.NiceRepr):
    """Stores ascend assignments between predicted and truth boxes.

    Attributes:
        concat_num_gts (list[int]): the number of truth boxes considered.

        concat_pos_mask (IntTensor): Positive samples mask in all images.

        concat_neg_mask (IntTensor): Negative samples mask in all images.

        concat_max_overlaps (FloatTensor): The max overlaps of all bboxes
            and ground truth boxes.

        concat_anchor_gt_indes(None | LongTensor): The the assigned truth
            box index of all anchors.

        concat_anchor_gt_labels(None | LongTensor): The gt labels
            of all anchors
    """

    def __init__(self,
                 concat_num_gts,
                 concat_pos_mask,
                 concat_neg_mask,
                 concat_max_overlaps,
                 concat_anchor_gt_indes=None,
                 concat_anchor_gt_labels=None):
        self.concat_num_gts = concat_num_gts
        self.concat_pos_mask = concat_pos_mask
        self.concat_neg_mask = concat_neg_mask
        self.concat_max_overlaps = concat_max_overlaps
        self.concat_anchor_gt_indes = concat_anchor_gt_indes
        self.concat_anchor_gt_labels = concat_anchor_gt_labels
        # Interface for possible user-defined properties
        self._extra_properties = {}
