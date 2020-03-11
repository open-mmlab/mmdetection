_base_ = [
    '../component/guided_anchoring/ga_faster_r50_caffe_fpn.py',
    '../component/coco_detection.py', '../component/schedule_1x.py',
    '../component/default_runtime.py'
]
model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        _delete_=True,
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))
work_dir = './work_dirs/ga_faster_rcnn_x101_32x4d_fpn_1x'
