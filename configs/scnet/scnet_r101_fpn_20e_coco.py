_base_ = './scnet_r50_fpn_20e_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
