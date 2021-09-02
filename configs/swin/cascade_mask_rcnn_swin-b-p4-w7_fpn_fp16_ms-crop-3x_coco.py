_base_ = './cascade_mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
pretrained = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth'  # noqa
model = dict(
    backbone=dict(embed_dims=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], drop_path_rate=0.3),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    init_cfg=dict(type='Pretrained', checkpoint=pretrained))
