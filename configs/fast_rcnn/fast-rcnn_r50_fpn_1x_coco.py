_base_ = [
    '../_base_/models/fast-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadProposals', num_max_proposals=2000),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='ProposalBroadcaster',
        transforms=[
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
        ]),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(
        type='ProposalBroadcaster',
        transforms=[
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    dataset=dict(
        proposal_file='proposals/rpn_r50_fpn_1x_train2017.pkl',
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        proposal_file='proposals/rpn_r50_fpn_1x_val2017.pkl',
        pipeline=test_pipeline))
test_dataloader = val_dataloader
