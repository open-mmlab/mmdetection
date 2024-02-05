_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth'  # noqa

model = dict(
    use_autocast=True,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=None),
    neck=dict(in_channels=[256, 512, 1024]),
)

o365v1_od_dataset = dict(
    type='ODVGDataset',
    data_root='data/objects365v1/',
    ann_file='o365v1_train_odvg.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None,
)

flickr30k_dataset = dict(
    type='ODVGDataset',
    data_root='data/flickr30k_entities/',
    ann_file='final_flickr_separateGT_train_vg.json',
    label_map_file=None,
    data_prefix=dict(img='flickr30k_images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None)

gqa_dataset = dict(
    type='ODVGDataset',
    data_root='data/gqa/',
    ann_file='final_mixed_train_no_coco_vg.json',
    label_map_file=None,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None)

v3d_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        # change this
        label_map_file='data/V3Det/annotations/v3det_2023_v1_label_map.json',
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]
v3det_dataset = dict(
    type='ODVGDataset',
    data_root='data/V3Det/',
    ann_file='annotations/v3det_2023_v1_train_od.json',
    label_map_file='annotations/v3det_2023_v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    need_text=False,  # change this
    pipeline=v3d_train_pipeline,
    return_classes=True,
    backend_args=None)

grit_dataset = dict(
    type='ODVGDataset',
    data_root='grit_processed/',
    ann_file='grit20m_vg.json',
    label_map_file=None,
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None)

# --------------------------- lvis od dataset---------------------------
lvis_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        # change this
        label_map_file='data/coco/annotations/lvis_v1_label_map.json',
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]
lvis_dataset = dict(
    type='ClassBalancedDataset',
    oversample_thr=1e-3,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='annotations/lvis_v1_train_od.json',
        label_map_file='annotations/lvis_v1_label_map.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False),
        need_text=False,  # change this
        pipeline=lvis_train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- coco2017 od dataset---------------------------
coco2017_train_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='annotations/instance_train2017_norefval_od.json',
        label_map_file='annotations/coco2017_label_map.json',
        data_prefix=dict(img='train2017'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- coco2014 vg dataset---------------------------
coco2014_vg_dataset = dict(
    type='ODVGDataset',
    data_root='data/coco/',
    ann_file='mdetr_annotations/final_mixed_train_only_coco_vg.json',
    label_map_file=None,
    data_prefix=dict(img='train2014/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None)

# --------------------------- refcoco vg dataset---------------------------
refcoco_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='mdetr_annotations/finetune_refcoco_train_vg.json',
        label_map_file=None,
        data_prefix=dict(img='train2014'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- refcoco+ vg dataset---------------------------
refcoco_plus_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='mdetr_annotations/finetune_refcoco+_train_vg.json',
        label_map_file=None,
        data_prefix=dict(img='train2014'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- refcocog vg dataset---------------------------
refcocog_dataset = dict(
    type='RepeatDataset',
    times=3,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='mdetr_annotations/finetune_refcocog_train_vg.json',
        label_map_file=None,
        data_prefix=dict(img='train2014'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- grefcoco vg dataset---------------------------
grefcoco_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='mdetr_annotations/finetune_grefcoco_train_vg.json',
        label_map_file=None,
        data_prefix=dict(img='train2014'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- dataloader---------------------------
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(
        _delete_=True,
        type='CustomSampleSizeSampler',
        ratio_mode=True,
        dataset_size=[-1, -1, 0.07, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
    dataset=dict(datasets=[
        o365v1_od_dataset,  # 1.74M
        v3det_dataset,  #
        grit_dataset,
        lvis_dataset,
        coco2017_train_dataset,  # 0.12M
        flickr30k_dataset,  # 0.15M
        gqa_dataset,  # 0.62M
        coco2014_vg_dataset,  # 0.49M
        refcoco_dataset,  # 0.12M
        refcoco_plus_dataset,  # 0.12M
        refcocog_dataset,  # 0.08M
        grefcoco_dataset,  # 0.19M
    ]))

optim_wrapper = dict(optimizer=dict(lr=0.0001))

# learning policy
max_iter = 304680
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=10000)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[228510],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=20))
log_processor = dict(by_epoch=False)
