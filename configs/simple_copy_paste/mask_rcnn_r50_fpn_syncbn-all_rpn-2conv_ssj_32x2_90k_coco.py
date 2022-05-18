_base_ = 'mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_270k_coco.py'

# lr steps at [0.9, 0.95, 0.975] of the maximum iterations
lr_config = dict(
    warmup_iters=500, warmup_ratio=0.067, step=[81000, 85500, 87750])
# 90k iterations with batch_size 64 is roughly equivalent to 48 epochs
runner = dict(type='IterBasedRunner', max_iters=90000)
