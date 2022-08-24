_base_ = './dino_5scale_swin-l-p4-w12_16x1_12e_coco.py'
lr_config = dict(step=[30])
runner = dict(max_epochs=36)
