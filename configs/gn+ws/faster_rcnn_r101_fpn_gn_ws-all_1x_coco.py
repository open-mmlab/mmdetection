_base_ = './faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py'
model = dict(
    pretrained='open-mmlab://jhu/resnet101_gn_ws', backbone=dict(depth=101))
