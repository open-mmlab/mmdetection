from ..builder import DETECTORS
from .cascade_rcnn import CascadeRCNN


@DETECTORS.register_module()
class HybridTaskCascade(CascadeRCNN):

    def __init__(self, **kwargs):
        super(HybridTaskCascade, self).__init__(**kwargs)

    @property
    def with_semantic(self):
        return self.roi_head.with_semantic
