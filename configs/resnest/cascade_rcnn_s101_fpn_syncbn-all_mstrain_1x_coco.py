_base_ = './cascade_rcnn_s50_fpn_syncbn-all_mstrain_1x_coco.py'
model = dict(
    pretrained='pretrain_model/resnest101-d6979a76.pth',
    backbone=dict(stem_channels=128, depth=101))
