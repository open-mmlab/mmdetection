_base_ = '../htc/htc_r50_fpn_1x_coco.py'
model = dict(
    pretrained='open-mmlab://res2net101_v1d_26w_4s',
    backbone=dict(type='Res2Net', depth=101, scales=4, base_width=26))
# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)
