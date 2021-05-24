_base_ = 'faster_rcnn_r50_fpn_mstrain_3x_coco.py'

model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
