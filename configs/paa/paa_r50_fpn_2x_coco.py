_base_ = './paa_r50_fpn_1x_coco.py'
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
