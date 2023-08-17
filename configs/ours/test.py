custom_imports = dict(imports=['mmseg.datasets', 'mmseg.models'], allow_failed_imports=False)

# custom_imports = dict(imports=['rssam.datasets', 'rssam.models'], allow_failed_imports=False)


# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# _base_ = [
#     '../_base_/default_runtime.py'
# ]



# default_scope = 'rssam'
default_scope = 'mmdet'

sub_model_train = [
    'panoptic_head',
    'data_preprocessor'
]

sub_model_optim = {
    'panoptic_head': {'lr_mult': 1},
}

# max_epochs = 1200
max_epochs = 100

# optimizer = dict(
#     type='AdamW',
#     sub_model=sub_model_optim,
#     lr=0.0005,
#     weight_decay=1e-3
# )

# optimizer = dict(type='AdamW', lr=0.0005, momentum=0.9, weight_decay=0.001)
# optimizer=dict(
#         type='AdamW', lr=0.00005, betas=(0.9, 0.999), weight_decay=0.001),
# optim_wrapper = dict(optimizer=optimizer)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.02 * 4, momentum=0.9, weight_decay=0.00004))


param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=1,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        by_epoch=True,
        begin=1,
        end=max_epochs,
    ),
]

param_scheduler_callback = dict(
    type='ParamSchedulerHook'
)

# from mmdet.evaluation import CocoMetric
# evaluator_ = dict(
#     type=CocoMetric,
#     metric=['bbox', 'segm'],
#     proposal_nums=[1, 10, 100]
# )

# evaluator = dict(
#     val_evaluator=evaluator_,
# )


image_size = (1024, 1024)

data_preprocessor = dict(
    type='mmdet.DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
)

num_things_classes = 10
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
prompt_shape = (60, 4)

from mmdet.models import SegSAMAnchor, SAMAnchorInstanceHead, SAMAggregatorNeck, SAMAnchorPromptRoIHead
# from mmdet.models.dense_heads import rpn_head
import mmdet

from mmdet.models.task_modules.assigners import MaxIoUAssigner

model = dict(
    type=SegSAMAnchor,
    # hyperparameters=dict(
    #     optimizer=optimizer,
    #     param_scheduler=param_scheduler,
    #     evaluator=evaluator,
    # ),
    need_train_names=sub_model_train,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='vit_h',
        checkpoint='/nfs/home/3002_hehui/xmx/pretrain/sam/sam_vit_h_4b8939.pth',
        # type='vit_b',
        # checkpoint='pretrain/sam/sam_vit_b_01ec64.pth',
    ),
    
    panoptic_head=dict(
        type=SAMAnchorInstanceHead,
        neck=dict(
            type=SAMAggregatorNeck,
            in_channels=[1280] * 32,
            # in_channels=[768] * 12,
            inner_channels=32,
            selected_channels=range(8, 32, 2),
            # selected_channels=range(4, 12, 2),
            out_channels=256,
            up_sample_scale=4,
        ),
        rpn_head=dict(
            type=mmdet.models.RPNHead,
            # type='mmdet.RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[2, 4, 8, 16, 32, 64],
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32]),
            bbox_coder=dict(
                type='mmdet.DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='mmdet.SmoothL1Loss', loss_weight=1.0)),
        roi_head=dict(
            type=SAMAnchorPromptRoIHead,
            bbox_roi_extractor=dict(
                type='mmdet.SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[8, 16, 32]),
            bbox_head=dict(
                type='mmdet.Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='mmdet.DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='mmdet.SmoothL1Loss', loss_weight=1.0)),
            mask_roi_extractor=dict(
                type='mmdet.SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[8, 16, 32]),
            mask_head=dict(
                type='SAMPromptMaskHead',
                per_query_point=prompt_shape[1],
                with_sincos=True,
                class_agnostic=True,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
        # model training and testing settings
        train_cfg=dict(

        rpn=dict(
            assigner=dict(
                type=MaxIoUAssigner,
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
        sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='mmdet.RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=1024,
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100,
                mask_thr_binary=0.5)
        )
    )
)


task_name = 'dev'
exp_name = 'rsprompter_anchor_nwpu-E20230722_0'
logger = dict(
    type='WandbLogger',
    project=task_name,
    group='sam-anchor',
    name=exp_name
)

# vis_backends = [dict(type='LocalVisBackend'), dict(type='WandBVisBackend')]
# visualizer = dict(vis_backends=vis_backends)

# visualizer = dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend')])

callbacks = [
    param_scheduler_callback,
    dict(
        type='ModelCheckpoint',
        dirpath=f'results/{task_name}/{exp_name}/checkpoints',
        save_last=True,
        mode='max',
        monitor='valsegm_map_0',
        save_top_k=3,
        filename='epoch_{epoch}-map_{valsegm_map_0:.4f}'
    ),
    dict(
        type='LearningRateMonitor',
        logging_interval='step'
    )
]


train_cfg = dict(
    by_epoch=True,
    val_interval=5,
    max_epochs=max_epochs,
    # compiled_model=False,
    # accelerator="auto",
    # strategy="auto",
    # strategy="ddp",
    # strategy='ddp_find_unused_parameters_true',
    # precision='32',
    # precision='16-mixed',
    # devices=1,
    # default_root_dir=f'results/{task_name}/{exp_name}',
    # default_root_dir='results/tmp',
    # max_epochs=max_epochs,
    # logger=logger,
    # callbacks=callbacks,
    # log_every_n_steps=5,
    # check_val_every_n_epoch=5,
    # benchmark=True,
    # sync_batchnorm=True,
    # fast_dev_run=True,

    # limit_train_batches=1,
    # limit_val_batches=0,
    # limit_test_batches=None,
    # limit_predict_batches=None,
    # overfit_batches=0.0,

    # val_check_interval=None,
    # num_sanity_val_steps=0,
    # enable_checkpointing=None,
    # enable_progress_bar=None,
    # enable_model_summary=None,
    # accumulate_grad_batches=32,
    # gradient_clip_val=15,
    # gradient_clip_algorithm='norm',
    # deterministic=None,
    # inference_mode: bool=True,
    # use_distributed_sampler=True,
    # profiler="simple",
    # detect_anomaly=False,
    # barebones=False,
    # plugins=None,
    # reload_dataloaders_every_n_epochs=0,
)


backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='mmdet.Resize', scale=image_size),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=image_size),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


train_batch_size_per_gpu = 2
train_num_workers = 2
test_batch_size_per_gpu = 2
test_num_workers = 2
persistent_workers = True

data_parent = '/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset'
train_data_prefix = ''
val_data_prefix = ''

from mmdet.datasets import NWPUInsSegDataset
dataset_type = NWPUInsSegDataset

val_loader = dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type=dataset_type,
            data_root=data_parent,
            ann_file='/nfs/home/3002_hehui/xmx/RS-SA/RSPrompter/data/NWPU/annotations/NWPU_instances_val.json',
            data_prefix=dict(img='positive image set'),
            test_mode=True,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=test_pipeline,
            backend_args=backend_args))

datamodule_cfg = dict(
    type='PLDataModule',
    train_loader=dict(
        batch_size=train_batch_size_per_gpu,
        num_workers=train_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type=dataset_type,
            data_root=data_parent,
            ann_file='/nfs/home/3002_hehui/xmx/RS-SA/RSPrompter/data/NWPU/annotations/NWPU_instances_train.json',
            data_prefix=dict(img='positive image set'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)
    ),
    val_loader=val_loader,
    # test_loader=val_loader
    predict_loader=val_loader
)

# from rssam.datasets import NWPUInsSegDataset


train_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
    type='NWPUInsSegDataset',
    data_root=data_parent,
    ann_file='/nfs/home/3002_hehui/xmx/RS-SA/RSPrompter/data/NWPU/annotations/NWPU_instances_train.json',
    data_prefix=dict(img='positive image set'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)
)

val_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_size=train_batch_size_per_gpu,
    num_workers=test_num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
    type='NWPUInsSegDataset',
    data_root=data_parent,
    ann_file='/nfs/home/3002_hehui/xmx/RS-SA/RSPrompter/data/NWPU/annotations/NWPU_instances_val.json',
    data_prefix=dict(img='positive image set'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)
)

val_loader = dict(
    batch_size=test_batch_size_per_gpu,
    num_workers=test_num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_parent,
        ann_file='/nfs/home/3002_hehui/xmx/RS-SA/RSPrompter/data/NWPU/annotations/NWPU_instances_val.json',
        data_prefix=dict(img='positive image set'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=test_pipeline,
        backend_args=backend_args)
)


test_dataloader = val_dataloader


data_root = data_parent

# CocoMetric
from mmdet.evaluation.metrics import CocoMetric

val_evaluator = dict(
    type=CocoMetric,
    ann_file='/nfs/home/3002_hehui/xmx/RS-SA/RSPrompter/data/NWPU/annotations/NWPU_instances_val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator


val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
