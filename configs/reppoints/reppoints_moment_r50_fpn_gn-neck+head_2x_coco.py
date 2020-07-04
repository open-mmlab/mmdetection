_base_ = './reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py'
lr_config = dict(step=[16, 22])
total_epochs = 24
