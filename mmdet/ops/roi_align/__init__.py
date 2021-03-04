from .roi_align import RoIAlign, roi_align
from mmcv import ops


ops.RoIAlign = RoIAlign
ops.roi_align = roi_align