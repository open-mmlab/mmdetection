_base_ = './dino_4scale_r50_8x2_12e_coco.py'
lr_config = dict(step=[20])
runner = dict(max_epochs=24)
