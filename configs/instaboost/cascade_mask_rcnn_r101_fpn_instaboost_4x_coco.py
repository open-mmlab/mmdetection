_base_ = './cascade_mask_rcnn_r50_fpn_instaboost_4x_coco.py'

model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
