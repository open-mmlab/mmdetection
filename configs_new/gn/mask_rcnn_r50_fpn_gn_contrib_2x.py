_base_ = [
    '../component/gn/mask_rcnn_r50_fpn_gn.py', '../component/coco_instance.py',
    '../component/schedule_2x.py', '../component/default_runtime.py'
]
model = dict(pretrained='open-mmlab://contrib/resnet50_gn')
work_dir = './work_dirs/mask_rcnn_r50_fpn_gn_contrib_2x'
