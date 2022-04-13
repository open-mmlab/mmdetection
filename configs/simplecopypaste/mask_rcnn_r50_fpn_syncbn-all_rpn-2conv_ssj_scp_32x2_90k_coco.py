_base_ = 'mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_270k_coco.py'

# For coco train2017 dataset,
# 1 epoch = batch_size16 * iterations7330
# batch_size64 * iterations90k ~ epoch48
# lr steps at [0.9, 0.95, 0.975] of the maximum iterations
lr_config = dict(
    warmup_iters=500, warmup_ratio=0.067, step=[81000, 85500, 87750])
runner = dict(type='IterBasedRunner', max_iters=90000)
