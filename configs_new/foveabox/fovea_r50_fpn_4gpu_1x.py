_base_ = [
    '../_base_/foveabox/fovea_r50_fpn.py', '../_base_/coco_detection.py',
    '../_base_/schedule_1x.py', '../_base_/default_runtime.py'
]
data = dict(imgs_per_gpu=4, workers_per_gpu=4)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
work_dir = './work_dirs/fovea_r50_fpn_4gpu_1x'
