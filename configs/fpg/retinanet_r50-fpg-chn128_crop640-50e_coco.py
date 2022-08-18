_base_ = 'retinanet_r50-fpg_crop640-50e_coco.py'

model = dict(
    neck=dict(out_channels=128, inter_channels=128),
    bbox_head=dict(in_channels=128))
