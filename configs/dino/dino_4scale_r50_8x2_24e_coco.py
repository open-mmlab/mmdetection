_base_ = ['./dino_4scale_r50_8x2_12e_coco.py']

lr_config = dict(policy='step', step=[20])
runner = dict(type='EpochBasedRunner', max_epochs=24)
