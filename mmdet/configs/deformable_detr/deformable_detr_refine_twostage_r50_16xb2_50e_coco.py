if '_base_':
    from .deformable_detr_refine_r50_16xb2_50e_coco import *

model.merge(dict(as_two_stage=True))
