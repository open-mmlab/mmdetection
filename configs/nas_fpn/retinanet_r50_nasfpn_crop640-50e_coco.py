_base_ = './retinanet_r50_fpn_crop640-50e_coco.py'

# model settings
model = dict(
    # `pad_size_divisor=128` ensures the feature maps sizes
    # in `NAS_FPN` won't mismatch.
    data_preprocessor=dict(pad_size_divisor=128),
    neck=dict(
        _delete_=True,
        type='NASFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        stack_times=7,
        start_level=1,
        norm_cfg=dict(type='BN', requires_grad=True)))
