_base_ = [
    '../component/guided_anchoring/ga_retinanet_r50_caffe_fpn.py',
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
        style='pytorch'))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
work_dir = './work_dirs/ga_retinanet_x101_32x4d_fpn_1x'
