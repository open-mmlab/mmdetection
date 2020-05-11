_base_ = './mask_rcnn_r50_fpn_gn-all_2x_coco.py'
model = dict(
    pretrained='open-mmlab://detectron/resnet101_gn', backbone=dict(depth=101))
