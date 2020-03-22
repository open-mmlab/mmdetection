_base_ = './mask_rcnn_r50_fpn_gn_2x.py'
model = dict(
    pretrained='open-mmlab://detectron/resnet101_gn', backbone=dict(depth=101))
work_dir = './work_dirs/mask_rcnn_r101_fpn_gn_2x'
