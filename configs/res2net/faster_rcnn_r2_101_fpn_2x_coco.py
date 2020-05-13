_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
model = dict(
    pretrained=  # NOQA
    'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s_mmdetv2-f0a600f9.pth',  # NOQA
    backbone=dict(type='Res2Net', depth=101, scale=4, base_width=26))
