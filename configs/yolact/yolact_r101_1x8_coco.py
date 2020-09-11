_base_ = './yolact_r50_1x8_coco.py'

model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
