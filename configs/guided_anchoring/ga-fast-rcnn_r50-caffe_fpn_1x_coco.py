_base_ = '../fast_rcnn/fast-rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    roi_head=dict(
        bbox_head=dict(bbox_coder=dict(target_stds=[0.05, 0.05, 0.1, 0.1]))),
    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(pos_iou_thr=0.6, neg_iou_thr=0.6, min_pos_iou=0.6),
            sampler=dict(num=256))),
    test_cfg=dict(rcnn=dict(score_thr=1e-3)))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=300),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]
# TODO: support loading proposals
data = dict(
    train=dict(
        proposal_file=data_root + 'proposals/ga_rpn_r50_fpn_1x_train2017.pkl',
        pipeline=train_pipeline),
    val=dict(
        proposal_file=data_root + 'proposals/ga_rpn_r50_fpn_1x_val2017.pkl',
        pipeline=test_pipeline),
    test=dict(
        proposal_file=data_root + 'proposals/ga_rpn_r50_fpn_1x_val2017.pkl',
        pipeline=test_pipeline))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
