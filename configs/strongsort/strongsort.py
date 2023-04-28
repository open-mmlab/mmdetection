_base_ = [
    '../bytetrack/bytetrack_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py',  # noqa: E501
]

model = dict(
    type='StrongSORT',
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/strongsort/mot_dataset/yolox_x_crowdhuman_mot17-private-half_20220812_192036-b6c9ce9a.pth'  # noqa: E501
        )),
    reid=dict(
        type='BaseReID',
        data_preprocessor=None,
        backbone=dict(
            type='mmcls.ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            num_classes=380,
            loss_cls=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
            loss_triplet=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth'  # noqa: E501
        )),
    cmc=dict(
        type='CameraMotionCompensation',
        warp_mode='cv2.MOTION_EUCLIDEAN',
        num_iters=100,
        stop_eps=0.00001),
    tracker=dict(
        _delete_=True,
        type='StrongSORTTracker',
        motion=dict(type='KalmanFilter', center_only=False, use_nsa=True),
        obj_score_thr=0.6,
        reid=dict(
            num_samples=None,
            img_scale=(256, 128),
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            match_score_thr=0.3,
            motion_weight=0.02,
        ),
        match_iou_thr=0.7,
        momentums=dict(embeds=0.1, ),
        num_tentatives=2,
        num_frames_retain=100),
    postprocess_model=dict(
        type='AppearanceFreeLink',
        checkpoint=  # noqa: E251
        'https://download.openmmlab.com/mmtracking/mot/strongsort/mot_dataset/aflink_motchallenge_20220812_190310-a7578ad3.pth',  # noqa: E501
        temporal_threshold=(0, 30),
        spatial_threshold=50,
        confidence_threshold=0.95,
    ))

train_dataloader = None
val_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False))
test_dataloader = val_dataloader

train_cfg = None

# evaluator
val_evaluator = dict(
    use_postprocess=True,
    postprocess_tracklet_cfg=[
        dict(
            type='InterpolateTracklets',
            min_num_frames=5,
            max_num_frames=20,
            use_gsi=True,
            smooth_tau=10)
    ])
test_evaluator = val_evaluator

default_hooks = dict(logger=dict(type='LoggerHook', interval=1))
del _base_.train_pipeline
del _base_.param_scheduler
del _base_.optim_wrapper
