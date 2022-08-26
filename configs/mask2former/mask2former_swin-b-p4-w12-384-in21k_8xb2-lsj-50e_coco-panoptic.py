_base_ = ['./mask2former_swin-b-p4-w12-384_8xb2-lsj-50e_coco-panoptic.py']
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth'  # noqa

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=pretrained)))
