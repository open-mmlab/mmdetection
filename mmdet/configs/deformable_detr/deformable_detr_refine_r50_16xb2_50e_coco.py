if '_base_':
    from .deformable_detr_r50_16xb2_50e_coco import *

model.merge(dict(with_box_refine=True))
