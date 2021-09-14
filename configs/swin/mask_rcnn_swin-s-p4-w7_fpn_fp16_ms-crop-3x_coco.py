_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
pretrained = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth'  # noqa
model = dict(
    backbone=dict(depths=[2, 2, 18, 2]),
    init_cfg=dict(type='Pretrained', checkpoint=pretrained))
