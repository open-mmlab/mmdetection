_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

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

data_root = 'data/flickr30k_entities/'

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        type='ODVGDataset',
        data_root=data_root,
        ann_file='flickr_simple_train_vg.json',
        data_prefix=dict(img='flickr30k_images/'),
        pipeline=test_pipeline,
        return_classes=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    outfile_path=data_root + 'flickr_simple_train_vg_v1.json',
    img_prefix=data_root + 'flickr30k_images/',
    score_thr=0.4,
    nms_thr=0.5,
    type='DumpODVGResults')
test_evaluator = val_evaluator
