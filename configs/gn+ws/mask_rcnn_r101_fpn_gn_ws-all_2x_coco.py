_base_ = './mask_rcnn_r50_fpn_gn_ws-all_2x_coco.py'
model = dict(
    pretrained='open-mmlab://jhu/resnet101_gn_ws', backbone=dict(depth=101))
