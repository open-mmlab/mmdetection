_base_ = './vfnet_r50_fpn_1x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
