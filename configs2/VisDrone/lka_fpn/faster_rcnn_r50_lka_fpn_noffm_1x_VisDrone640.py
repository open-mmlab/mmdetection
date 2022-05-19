_base_ = './faster_rcnn_r50_lka_fpn_1x_VisDrone640.py'

model = dict(
    neck=dict(
        type='lka_FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        with_aem=False,
        with_ffm=True),
)