_base_ = [
    'retinanet_pvt_t_fpn_1x_coco.py',
]
model = dict(
    pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_large.pth',
    backbone=dict(num_layers=[3, 8, 27, 3]))
# optimizer
# The learning rate is divided by sqrt(2) because the batch size is reduced from 16 to 8.
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001 / 1.4, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# dataset settings
# Due to GPU memory limitations, the batch size was reduced from 16 to 8.
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1)