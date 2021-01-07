_base_ = ['../vfnet/vfnet_r50_fpn_1x_coco.py', '../_base_/swa.py']
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
swa_optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
