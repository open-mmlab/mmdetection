_base_ = [
    '../common/mstrain_3x_coco.py', '../_base_/models/faster_rcnn_r50_fpn.py'
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4
)