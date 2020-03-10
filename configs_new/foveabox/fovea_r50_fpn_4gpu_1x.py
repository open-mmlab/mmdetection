_base_ = [
    '../component/foveabox/fovea_r50_fpn.py', '../component/coco_detection.py',
    '../component/schedule_1x.py', '../component/default_runtime.py'
]
data = dict(imgs_per_gpu=4, workers_per_gpu=4)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
work_dir = './work_dirs/fovea_r50_fpn_4gpu_1x'
