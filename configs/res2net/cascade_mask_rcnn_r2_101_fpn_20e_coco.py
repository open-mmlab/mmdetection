_base_ = '../cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco.py'
model = dict(
    pretrained=  # NOQA
    'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',  # NOQA
    backbone=dict(type='Res2Net', depth=101, scale=4, baseWidth=26))
