_base_ = './dino_4scale_swin-l-p4-w12_8x2_12e_coco.py'
lr_config = dict(step=[30])
runner = dict(max_epochs=36)
