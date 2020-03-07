_base_ = [
    'component/faster_rcnn_r50_fpn.py', 'component/coco_detection.py',
    'component/schedule_1x.py', 'component/default_runtime.py'
]
work_dir = './work_dirs/faster_rcnn_r50_fpn_1x'
