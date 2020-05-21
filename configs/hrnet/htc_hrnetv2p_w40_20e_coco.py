_base_ = './htc_hrnetv2p_w32_20e_coco.py'
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w40',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage2=dict(num_channels=(40, 80)),
            stage3=dict(num_channels=(40, 80, 160)),
            stage4=dict(num_channels=(40, 80, 160, 320)))),
    neck=dict(type='HRFPN', in_channels=[40, 80, 160, 320], out_channels=256))
