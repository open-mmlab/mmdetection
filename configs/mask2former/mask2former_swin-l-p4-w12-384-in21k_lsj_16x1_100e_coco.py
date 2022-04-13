_base_ = ['./mask2former_swin-b-p4-w12-384_lsj_8x2_50e_coco.py']
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa

model = dict(
    backbone=dict(
        embed_dims=192,
        num_heads=[6, 12, 24, 48],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    panoptic_head=dict(num_queries=200, in_channels=[192, 384, 768, 1536]))

data = dict(samples_per_gpu=1, workers_per_gpu=1)

lr_config = dict(step=[655556, 710184])

max_iters = 737500
runner = dict(type='IterBasedRunner', max_iters=max_iters)

# Before 735001th iteration, we do evaluation every 5000 iterations.
# After 735000th iteration, we do evaluation every 737500 iterations,
# which means that we do evaluation at the end of training.'
interval = 5000
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
evaluation = dict(
    interval=interval,
    dynamic_intervals=dynamic_intervals,
    metric=['PQ', 'bbox', 'segm'])
