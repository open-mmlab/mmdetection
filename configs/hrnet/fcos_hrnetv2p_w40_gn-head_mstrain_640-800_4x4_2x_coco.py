_base_ = './fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco.py'
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w40',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage2=dict(num_channels=(40, 80)),
            stage3=dict(num_channels=(40, 80, 160)),
            stage4=dict(num_channels=(40, 80, 160, 320)))),
    neck=dict(type='HRFPN', in_channels=[40, 80, 160, 320], out_channels=256))
