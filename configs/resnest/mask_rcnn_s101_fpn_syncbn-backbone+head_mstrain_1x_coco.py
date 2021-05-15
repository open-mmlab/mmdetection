_base_ = './mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py'
model = dict(
    pretrained='open-mmlab://resnest101',
    backbone=dict(stem_channels=128, depth=101))
