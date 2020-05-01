_base_ = './grid_rcnn_r50_fpn_gn-head_2x_coco.py'

model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
