_base_ = [
    'component/fast_rcnn_r50_fpn.py', 'component/coco_proposal_detection.py',
    'component/schedule_1x.py', 'component/default_runtime.py'
]
work_dir = './work_dirs/fast_rcnn_r50_fpn_1x'
