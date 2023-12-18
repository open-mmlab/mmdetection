_base_ = '../grounding_dino/grounding_dino_swin-b_pretrain_mixeddata.py'

model = dict(test_cfg=dict(max_per_img=10))

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadTextAnnotations'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

data_root = 'data/'

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        type='ODVGDataset',
        data_root=data_root,
        ann_file='final_flickr_separateGT_train_vg.json',
        data_prefix=dict(img='flickr30k_images/'),
        pipeline=test_pipeline,
        return_classes=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    outfile_path='aa.json',
    img_prefix=data_root + 'flickr30k_images/',
    type='DumpODVGResults')
test_evaluator = val_evaluator
