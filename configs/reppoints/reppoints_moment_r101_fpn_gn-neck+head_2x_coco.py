_base_ = './reppoints_moment_r50_fpn_gn-neck+head_2x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
