from ..geometry import bbox_overlaps
from .registry import IOU_CALCULATOR


@IOU_CALCULATOR.register_module
class BboxOverlaps2D(object):
    """2D IoU Calculator"""

    def __init__(self, mode='iou', is_aligned=False):
        self.mode = mode
        self.is_aligned = is_aligned

    def __call__(self, bboxes1, bboxes2, mode=None, is_aligned=None):
        if mode is None:
            mode = self.mode
        if is_aligned is None:
            is_aligned = self.is_aligned
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mode={}, is_aligned={})'.format(self.mode,
                                                      self.is_aligned)
        return repr_str
