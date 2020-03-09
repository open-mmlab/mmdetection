_base_ = [
    'component/retinanet_r50_fpn.py', 'component/coco_detection.py',
    'component/schedule_1x.py', 'component/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
work_dir = './work_dirs/retinanet_r50_fpn_1x'
