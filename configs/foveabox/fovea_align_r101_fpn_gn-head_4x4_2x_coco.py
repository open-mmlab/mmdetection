_base_ = './fovea_r50_fpn_4x4_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    bbox_head=dict(
        with_deform=True,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)))
# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
