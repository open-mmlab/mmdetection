_base_ = './mask_rcnn_x101_32x4d_fpn_1x.py'
model = dict(
    pretrained='open-mmlab://resnext101_64x4d', backbone=dict(groups=64, ))
work_dir = './work_dirs/mask_rcnn_x101_64x4d_fpn_1x'
