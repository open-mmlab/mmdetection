_base_ = './htc_r50_fpn_1x_coco.py'
# learning policy
lr_config = dict(step=[16, 19])
total_epochs = 20
work_dir = './work_dirs/htc_r50_fpn_20e'
