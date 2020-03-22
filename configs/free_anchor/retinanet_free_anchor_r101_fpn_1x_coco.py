_base_ = './retinanet_free_anchor_r50_fpn_1x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
work_dir = './work_dirs/retinanet_free_anchor_r101_fpn_1x'
