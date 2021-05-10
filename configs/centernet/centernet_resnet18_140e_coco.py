_base_ = './centernet_resnet18_dcnv2_140e_coco.py'

model = dict(neck=dict(use_dcn=False))
