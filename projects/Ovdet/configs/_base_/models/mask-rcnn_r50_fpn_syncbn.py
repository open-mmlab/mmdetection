_base_ = 'mmdet::_base_/models/mask-rcnn_r50_fpn.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
# model settings
model = dict(
    backbone=dict(
        frozen_stages=-1, norm_cfg=norm_cfg, norm_eval=False, init_cfg=None),
    neck=dict(norm_cfg=norm_cfg, ),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            norm_cfg=norm_cfg,
            norm_eval=False,
            num_classes=80,
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CustomCrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
        ),
        mask_head=dict(
            norm_cfg=norm_cfg,
            class_agnostic=True,
        )),
    # model training and testing settings
    test_cfg=dict(rcnn=dict(score_thr=0.05, )))
