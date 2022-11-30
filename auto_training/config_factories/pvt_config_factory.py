#spec_base_path = "configs/auto/auto.py"
import pdb

from mmcv import Config
from auto_training.utils.utils import parse_training_data_classes


def make_pvt_cfg(data_path: str,
                 input_res = (512, 384),
                 work_dir='./work_dirs/pvtb0',
                 keep_ratio = False):

    cfg = Config()
    cfg.work_dir = work_dir
    cfg.dataset_type = 'AutoDataset'
    cfg.data_root = data_path
    cfg.train_ann_file = f"{cfg.data_root}/train/coco_train.json"
    cfg.train_img_prefix = f"{cfg.data_root}/train/image_2"
    cfg.val_ann_file = f"{cfg.data_root}/val/coco_val.json"
    cfg.val_img_prefix = f"{cfg.data_root}/val/image_2/"

    min_res = (input_res[0], input_res[1] * 0.8)
    training_classes = parse_training_data_classes(cfg.train_ann_file)
    num_classes = len(training_classes)
    cfg.input_res = input_res
    cfg.used_classes = [cls["name"] for cls in training_classes]
    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=[input_res, min_res], keep_ratio=keep_ratio),
        # dict(type='Rot90'),
        dict(
            type='RandomCrop',
            crop_type='relative_range',
            crop_size=(0.7, 1.0),
            allow_negative_crop=True),
        dict(type='RandomAffine'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]
    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=input_res,
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=keep_ratio),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ]
    cfg.data = dict(
        samples_per_gpu=4,
        workers_per_gpu=2,
        train=dict(
            type='RepeatDataset',
            times=8,
            dataset=dict(
                type=cfg.dataset_type,
                ann_file=
                cfg.train_ann_file,
                img_prefix=
                cfg.train_img_prefix,
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='Resize',
                        img_scale=[input_res, min_res],
                        keep_ratio=keep_ratio),
                    # dict(type='Rot90'),
                    dict(
                        type='RandomCrop',
                        crop_type='relative_range',
                        crop_size=(0.7, 1.0),
                        allow_negative_crop=True),
                    dict(type='RandomAffine'),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ])),
        val=dict(
            type=cfg.dataset_type,
            ann_file=cfg.val_ann_file,
            img_prefix=cfg.val_img_prefix,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=input_res,
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=keep_ratio),
                        dict(type='RandomFlip'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='Pad', size_divisor=32),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ]),
        test=dict(
            type=cfg.dataset_type,
            ann_file=cfg.val_ann_file,
            img_prefix=cfg.val_img_prefix,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=input_res,
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=keep_ratio),
                        dict(type='RandomFlip'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='Pad', size_divisor=32),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ]))
    cfg.evaluation = dict(interval=1, metric='bbox')
    cfg.optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
    cfg.optimizer_config = dict(grad_clip=None)
    cfg.lr_config = dict(
        policy='step',
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=0.001,
        step=[16, 22])
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=24)
    cfg.checkpoint_config = dict(interval=1)
    cfg.log_config = dict(
        interval=50,
        hooks=[dict(type='TextLoggerHook'),
               dict(type='TensorboardLoggerHook')])
    cfg.custom_hooks = [dict(type='NumClassCheckHook')]
    cfg.dist_params = dict(backend='nccl')
    cfg.log_level = 'INFO'
    cfg.load_from = None
    cfg.resume_from = None
    cfg.workflow = [('train', 1)]
    cfg.opencv_num_threads = 0
    cfg.mp_start_method = 'fork'
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=16)
    cfg.auto_resume = False
    cfg.gpu_ids = [0]
    cfg.model = dict(
        type='RetinaNet',
        backbone=dict(
            type='PyramidVisionTransformerV2',
            embed_dims=32,
            num_layers=[2, 2, 2, 2],
            init_cfg=dict(
                checkpoint=
                'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth'
            )),
        neck=dict(
            type='FPN',
            in_channels=[32, 64, 160, 256],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_input',
            num_outs=5),
        bbox_head=dict(
            type='RetinaHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))
    return cfg