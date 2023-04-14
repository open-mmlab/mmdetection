if '_base_':
    from .crowddet_rcnn_r50_fpn_8xb2_30e_crowdhuman import *

model.merge(dict(roi_head=dict(bbox_head=dict(with_refine=True))))
