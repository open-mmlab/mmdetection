dataset_type = 'ADE20KSemsegDataset'
data_root = 'data/ade/ADEChallengeData2016'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadSemSegAnnotations'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'seg_map_path', 'img',
                   'gt_seg_map', 'text'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='SemSegMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
