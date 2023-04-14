if '_base_':
    from .centernet_r18_dcnv2_8xb16_crop512_140e_coco import *

model.merge(dict(neck=dict(use_dcn=False)))
