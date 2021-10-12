dataset_type = 'CocoFmtDataset'
data_root = 'data/tiny_set/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', scale_factor=[1.0], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        # type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        type='CroppedTilesFlipAug',
        tile_shape=(640, 512),  # sub image size by cropped
        tile_overlap=(100, 100),
        scale_factor=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
            type=dataset_type,
            # ann_file=data_root + 'erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json',
            ann_file=data_root + 'mini_annotations/tiny_set_train_sw640_sh512_all_erase.json',  # same as last line
            img_prefix=data_root + 'erase_with_uncertain_dataset/train/',
            pipeline=train_pipeline,
            # train_ignore_as_bg=False,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'mini_annotations/tiny_set_test_all.json',
        # ann_file=data_root + 'annotations/corner/task/tiny_set_test_sw640_sh512_all.json',
        # merge_after_infer_kwargs=dict(
        #     merge_gt_file=data_root + 'mini_annotations/tiny_set_test_all.json',
        #     merge_nms_th=0.5
        # ),
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/corner/task/tiny_set_test_sw640_sh512_all.json',
        ann_file=data_root + 'mini_annotations/tiny_set_test_all.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline)
)

check = dict(stop_while_nan=True)  # add by hui

# origin coco eval
# evaluation = dict(interval=4, metric='bbox')

# tiny bbox eval with IOD
evaluation = dict(
    interval=3, metric='bbox',
    iou_thrs=[0.25, 0.5, 0.75],  # set None mean use 0.5:1.0::0.05
    proposal_nums=[1000],
    cocofmt_kwargs=dict(
        ignore_uncertain=True,
        use_ignore_attr=True,
        use_iod_for_ignore=True,
        iod_th_of_iou_f="lambda iou: iou",  #"lambda iou: (2*iou)/(1+iou)",
        cocofmt_param=dict(
            evaluate_standard='tiny',  # or 'coco'
            # iouThrs=[0.25, 0.5, 0.75],  # set this same as set evaluation.iou_thrs
            # maxDets=[200],              # set this same as set evaluation.proposal_nums
        )
    )
)

# location bbox eval
# evaluation = dict(
#     interval=4, metric='bbox',
#     use_location_metric=True,
#     location_kwargs=dict(
#         matcher_kwargs=dict(multi_match_not_false_alarm=False),
#         location_param=dict(
#             matchThs=[0.5, 1.0, 2.0],
#             recThrs='np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)',
#             maxDets=[300],
#             # recThrs='np.linspace(.90, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)',
#             # maxDets=[1000],
#         )
#     )
# )
