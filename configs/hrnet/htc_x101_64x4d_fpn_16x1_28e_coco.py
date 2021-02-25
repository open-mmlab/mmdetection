_base_ = '../htc/htc_x101_64x4d_fpn_16x1_20e_coco.py'
# learning policy
lr_config = dict(step=[24, 27])
runner = dict(type='EpochBasedRunner', max_epochs=28)
