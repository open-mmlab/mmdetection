_base_ = './vfnet_r50_fpn_mstrain_2x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
