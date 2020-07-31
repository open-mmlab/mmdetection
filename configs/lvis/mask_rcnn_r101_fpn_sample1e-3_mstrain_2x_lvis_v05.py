_base_ = './mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_v05.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
