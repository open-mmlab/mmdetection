_base_ = './centernet_r18_dcnv2_140e_coco.py'

model = dict(neck=dict(use_dcn=False))
