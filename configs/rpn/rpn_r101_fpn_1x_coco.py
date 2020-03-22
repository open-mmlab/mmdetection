_base_ = './rpn_r50_fpn_1x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
work_dir = './work_dirs/rpn_r101_fpn_1x'
