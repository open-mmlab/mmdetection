_base_ = '../lka_fpn/faster_rcnn_r50_lka_fpn_1x_TinyPerson640_newData.py'


model=dict(
    neck=dict(
        type='lka_FPN_ssm',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5)
)