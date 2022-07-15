_base_ = 'mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_270k_coco.py'

# training schedule for 90k
max_iters = 90000

# learning rate policy
# lr steps at [0.9, 0.95, 0.975] of the maximum iterations
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.067, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=90000,
        by_epoch=False,
        milestones=[81000, 85500, 87750],
        gamma=0.1)
]
