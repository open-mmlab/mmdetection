_base_ = [
    '../_base_/models/faster-rcnn_r50-caffe-c4.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='TridentFasterRCNN',
    backbone=dict(
        type='TridentResNet',
        trident_dilations=(1, 2, 3),
        num_branch=3,
        test_branch_idx=1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    roi_head=dict(type='TridentRoIHead', num_branch=3, test_branch_idx=1),
    train_cfg=dict(
        rpn_proposal=dict(max_per_img=500),
        rcnn=dict(
            sampler=dict(num=128, pos_fraction=0.5,
                         add_gt_as_proposals=False))))
