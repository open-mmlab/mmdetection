_base_ = './coco_sparse_rcnn_r50_fpn_mstrain_480-800_3x.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))