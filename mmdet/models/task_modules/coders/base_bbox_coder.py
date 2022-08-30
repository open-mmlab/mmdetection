# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class BaseBBoxCoder(metaclass=ABCMeta):
    """Base bounding box coder."""

    # The length of the `encode` function output.
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
