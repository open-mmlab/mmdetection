_base_ = [
    'component/cascade_rcnn_r50_fpn.py', 'component/coco_detection.py',
    'component/schedule_1x.py', 'component/default_runtime.py'
]
work_dir = './work_dirs/cascade_rcnn_r50_fpn_1x'
