_base_ = [
    '../_base_/models/retinanet_r50-fpn.py', '../common/mstrain_3x_coco.py'
]
# optimizer
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
