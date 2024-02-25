_base_ = [
    'mmdet::dino/dino-4scale_r50_8xb2-12e_coco.py'
]

# please install the mmclassification dev-1.x branch
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-atto_fcmae-pre_3rdparty_in1k_20230104-23765f83.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.ConvNeXt',
        arch='atto',
        out_indices=[1, 2, 3],
        drop_path_rate=0.1,
        layer_scale_init_value=0.,
        gap_before_final_norm=False,
        use_grn=True,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[80, 160, 320]))
