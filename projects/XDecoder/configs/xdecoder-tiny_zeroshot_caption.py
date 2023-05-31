_base_ = 'xdecoder-tiny_zeroshot_open-vocab-semseg.py'

model = dict(task='caption')

test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow', backend_args=_base_.backend_args),
    dict(type='FixScaleResize',
         scale=224,
         keep_ratio=True,
         short_side_mode=True,
         backend='pillow',
         interpolation='bicubic'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dataset_type = 'COCOCaptionDataset'
data_root = 'data/coco/'

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/coco_karpathy_val.json',
        pipeline=test_pipeline,
    ))

val_evaluator = dict(
    type='COCOCaptionMetric',
    ann_file=data_root + 'annotations/coco_karpathy_val_gt.json',
)

# # If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
