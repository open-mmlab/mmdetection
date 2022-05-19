_base_ = '../base/faster_rcnn_r50_fpn_1x_VisDrone640.py'

model = dict(
    neck=dict(
        type='FPN_CARAFE',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        start_level=0,
        end_level=-1,
        norm_cfg=None,
        act_cfg=None,
        order=('conv', 'norm', 'act'),
        upsample_cfg=dict(
            type='carafe_pytorch',
            k_up=5,
            k_enc=3,
            c_mid=64))
)