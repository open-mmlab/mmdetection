_base_ = './rpn_r50_fpn_1x_coco.py'

# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24
