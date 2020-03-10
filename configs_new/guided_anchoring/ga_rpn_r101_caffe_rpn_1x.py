_base_ = './ga_rpn_r50_caffe_fpn_1x.py'
# model settings
model = dict(
    pretrained='open-mmlab://resnet101_caffe',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'))
work_dir = './work_dirs/ga_rpn_r101_caffe_fpn_1x'
