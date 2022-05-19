_base_ = '../PAFPN/faster_rcnn_r101_pafpn_1x_TinyPerson640_newData.py'

model=dict(
    neck=dict(type='PAFPN_LKAATTENTION_UNIFIED_CARAFE',
              in_channels=[256, 512, 1024, 2048],
              out_channels=256,
              num_outs=5)
)