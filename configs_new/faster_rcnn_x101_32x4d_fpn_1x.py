_base_ = './faster_rcnn_r101_fpn_1x.py'
model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        groups=32,
        base_width=4,
    ))
work_dir = './work_dirs/faster_rcnn_x101_32x4d_fpn_1x'
