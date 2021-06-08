_base_ = './mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
