_base_ = 'retinanet_r50_fpg_crop640_50e_coco.py'

model = dict(
    neck=dict(out_channels=128, inter_channels=128),
    bbox_head=dict(in_channels=128))
