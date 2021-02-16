_base_ = './htc_r50_fpn_1x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
# learning policy
lr_config = dict(step=[16, 19])
runner = dict(max_epochs=20)
