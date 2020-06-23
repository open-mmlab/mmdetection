_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
model = dict(
    pretrained='open-mmlab://res2net101_v1d_26w_4s',
    backbone=dict(type='Res2Net', depth=101, scales=4, base_width=26))
