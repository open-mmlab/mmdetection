_base_ = '../PAFPN/faster_rcnn_r50_pafpn_1x_VisDrone640.py'

model=dict(
    neck=dict(type='PAFPN_LKAATTENTION',
              in_channels=[256, 512, 1024, 2048],
              out_channels=256,
              num_outs=5)
)