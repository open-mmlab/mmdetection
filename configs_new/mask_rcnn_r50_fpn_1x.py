_base_ = [
    'component/mask_rcnn_r50_fpn.py', 'component/coco_instance.py',
    'component/schedule_1x.py', 'component/default_runtime.py'
]
work_dir = './work_dirs/mask_rcnn_r50_fpn_1x'
