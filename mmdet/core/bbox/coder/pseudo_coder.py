from ..registry import BBOX_CODER
from .base_coder import BaseCoder


@BBOX_CODER.register_module
class PseudoCoder(BaseCoder):

    def __init__(self, **kwargs):
        super(BaseCoder, self).__init__(**kwargs)

    def encode(self, bboxes, gt_bboxes):
        return gt_bboxes

    def decode(self, bboxes, pred_bboxes):
        return pred_bboxes
