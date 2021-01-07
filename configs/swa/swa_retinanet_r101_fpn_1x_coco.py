_base_ = ['../retinanet/retinanet_r50_fpn_1x_coco.py', '../_base_/swa.py']
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
