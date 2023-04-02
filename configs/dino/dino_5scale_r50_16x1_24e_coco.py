_base_ = './dino_5scale_r50_16x1_12e_coco.py'
lr_config = dict(step=[20])
runner = dict(max_epochs=24)
