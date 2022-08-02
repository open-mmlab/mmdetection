_base_ = [
    '../_base_/models/fast_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadProposals', num_max_proposals=2000),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='TransformBroadcaster',
        mapping=dict(
            # process `gt_bboxes` and `proposals` with same transforms
            gt_bboxes=['gt_bboxes', 'proposals'],
            # necessary keys that need update during processing
            # TODO: enhance logic in mmengine or mmdet
            ori_shape=['ori_shape', ...],
            img_shape=['img_shape', ...],
            scale_factor=['scale_factor', ...],
            flip=['flip', ...],
            flip_direction=['flip_direction', ...]),
        share_random_params=True,
        allow_nonexist_keys=True,
        auto_remap=True,
        transforms=[
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
        ]),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(
        type='TransformBroadcaster',
        mapping=dict(
            # process `gt_bboxes` and `proposals` with same transforms
            gt_bboxes=['gt_bboxes', 'proposals'],
            # necessary keys that need update during processing
            # TODO: enhance logic in mmengine or mmdet
            ori_shape=['ori_shape', ...],
            img_shape=['img_shape', ...],
            scale_factor=['scale_factor', ...],
        ),
        share_random_params=True,
        allow_nonexist_keys=True,
        auto_remap=True,
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
