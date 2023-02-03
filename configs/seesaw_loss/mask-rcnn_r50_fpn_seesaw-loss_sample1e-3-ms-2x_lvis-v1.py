_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v1_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1203,
            cls_predictor_cfg=dict(type='NormedLinear', tempearture=20),
            loss_cls=dict(
                type='SeesawLoss',
                p=0.8,
                q=2.0,
                num_classes=1203,
                loss_weight=1.0)),
        mask_head=dict(num_classes=1203)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_dataloader = dict(dataset=dict(dataset=dict(pipeline=train_pipeline)))

train_cfg = dict(val_interval=24)
