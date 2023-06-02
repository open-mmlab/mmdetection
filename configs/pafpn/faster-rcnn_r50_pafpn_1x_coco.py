_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

model = dict(
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5))
