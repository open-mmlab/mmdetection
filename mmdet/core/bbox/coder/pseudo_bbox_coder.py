from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class PseudoBBoxCoder(BaseBBoxCoder):

    def __init__(self, **kwargs):
        super(BaseBBoxCoder, self).__init__(**kwargs)

    def encode(self, bboxes, gt_bboxes):
        return gt_bboxes

    def decode(self, bboxes, pred_bboxes):
        return pred_bboxes
