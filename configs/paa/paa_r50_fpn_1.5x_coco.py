_base_ = './paa_r50_fpn_1x_coco.py'
lr_config = dict(step=[12, 16])
total_epochs = 18
