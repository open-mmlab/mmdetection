from ..registry import BBOX_CODER
from ..transforms import bbox2delta, delta2bbox
from .base_coder import BaseCoder


@BBOX_CODER.register_module
class DeltaCoder(BaseCoder):

    def __init__(self,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 **kwargs):
        super(BaseCoder, self).__init__(**kwargs)
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(1) == gt_bboxes.size(1) == 4
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    max_shape, wh_ratio_clip)

        return decoded_bboxes
