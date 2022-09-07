# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class BaseBBoxCoder(metaclass=ABCMeta):
    """Base bounding box coder.

    Args:
        with_boxlist (bool): Whether to warp decoded boxes with the
            boxlist data structure. Defaults to False.
    """

    # The size of the last of dimension of the encoded tensor.
    encode_size = 4

    def __init__(self, with_boxlist: bool = False, **kwargs):
        self.with_boxlist = with_boxlist

    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
        """Encode deltas between bboxes and ground truth boxes."""

    @abstractmethod
    def decode(self, bboxes, bboxes_pred):
        """Decode the predicted bboxes according to prediction and base
        boxes."""
