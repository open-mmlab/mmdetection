_base_ = ['retinanet_pvt_v2_b0_fpn_1x_coco.py']
model = dict(
    pretrained='https://github.com/whai362/PVT/'
    'releases/download/v2/pvt_v2_b5.pth',
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 6, 40, 3],
        mlp_ratios=(4, 4, 4, 4),
    ),
    neck=dict(in_channels=[64, 128, 320, 512]))
# optimizer
# The learning rate is divided by sqrt(2) because
# the batch size is reduced from 16 to 8.
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001 / 1.4, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# dataset settings
# Due to GPU memory limitations, the batch size
# was reduced from 16 to 8.
data = dict(samples_per_gpu=1, workers_per_gpu=1)
