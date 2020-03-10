_base_ = [
    '../component/mask_rcnn_r50_fpn.py', '../component/coco_instance.py',
    '../component/schedule_1x.py', '../component/default_runtime.py'
]
model = dict(
    type='MaskScoringRCNN',
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    mask_iou_head=dict(
        type='MaskIoUHead',
        num_convs=4,
        num_fcs=2,
        roi_feat_size=14,
        in_channels=256,
        conv_out_channels=256,
        fc_out_channels=1024,
        num_classes=81))
# model training and testing settings
train_cfg = dict(rcnn=dict(mask_thr_binary=0.5))
work_dir = './work_dirs/ms_rcnn_x101_64x4d_fpn_1x'
