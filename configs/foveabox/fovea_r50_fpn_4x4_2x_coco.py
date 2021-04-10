_base_ = './fovea_r50_fpn_4x4_1x_coco.py'
# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
