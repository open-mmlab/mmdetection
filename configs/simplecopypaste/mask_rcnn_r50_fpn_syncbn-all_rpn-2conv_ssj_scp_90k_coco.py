_base_ = 'mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_270k_coco.py'

lr_config = dict(
    warmup_iters=500, warmup_ratio=0.067, step=[81000, 85500, 87750])

runner = dict(type='IterBasedRunner', max_iters=90000)
