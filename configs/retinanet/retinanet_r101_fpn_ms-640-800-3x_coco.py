_base_ = ['../_base_/models/retinanet_r50_fpn.py', '../common/ms_3x_coco.py']
# optimizer
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
