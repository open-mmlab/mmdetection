_base_ = ['../grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py']
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# runtime settings
total_epochs = 12
