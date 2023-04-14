if '_base_':
    from .._base_.models.cascade_mask_rcnn_r50_fpn import *
    from .._base_.datasets.coco_instance import *
    from .._base_.schedules.schedule_2x import *
    from .._base_.default_runtime import *
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead, Shared2FCBBoxHead, Shared2FCBBoxHead
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder, DeltaXYWHBBoxCoder, DeltaXYWHBBoxCoder
from mmdet.models.layers.normed_predictor import NormedLinear, NormedLinear, NormedLinear
from mmdet.models.losses.seesaw_loss import SeesawLoss, SeesawLoss, SeesawLoss
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss, SmoothL1Loss, SmoothL1Loss
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmcv.transforms.processing import RandomChoiceResize
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.evaluation.metrics.lvis_metric import LVISMetric

model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101')),
        roi_head=dict(
            bbox_head=[
                dict(
                    type=Shared2FCBBoxHead,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=1203,
                    bbox_coder=dict(
                        type=DeltaXYWHBBoxCoder,
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]),
                    reg_class_agnostic=True,
                    cls_predictor_cfg=dict(type=NormedLinear, tempearture=20),
                    loss_cls=dict(
                        type=SeesawLoss,
                        p=0.8,
                        q=2.0,
                        num_classes=1203,
                        loss_weight=1.0),
                    loss_bbox=dict(
                        type=SmoothL1Loss, beta=1.0, loss_weight=1.0)),
                dict(
                    type=Shared2FCBBoxHead,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=1203,
                    bbox_coder=dict(
                        type=DeltaXYWHBBoxCoder,
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.05, 0.05, 0.1, 0.1]),
                    reg_class_agnostic=True,
                    cls_predictor_cfg=dict(type=NormedLinear, tempearture=20),
                    loss_cls=dict(
                        type=SeesawLoss,
                        p=0.8,
                        q=2.0,
                        num_classes=1203,
                        loss_weight=1.0),
                    loss_bbox=dict(
                        type=SmoothL1Loss, beta=1.0, loss_weight=1.0)),
                dict(
                    type=Shared2FCBBoxHead,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=1203,
                    bbox_coder=dict(
                        type=DeltaXYWHBBoxCoder,
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.033, 0.033, 0.067, 0.067]),
                    reg_class_agnostic=True,
                    cls_predictor_cfg=dict(type=NormedLinear, tempearture=20),
                    loss_cls=dict(
                        type=SeesawLoss,
                        p=0.8,
                        q=2.0,
                        num_classes=1203,
                        loss_weight=1.0),
                    loss_bbox=dict(
                        type=SmoothL1Loss, beta=1.0, loss_weight=1.0))
            ],
            mask_head=dict(num_classes=1203)),
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.0001,
                # LVIS allows up to 300
                max_per_img=300))))

# dataset settings
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True),
    dict(
        type=RandomChoiceResize,
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis_v1/'
train_dataloader.merge(
    dict(
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/lvis_v1_train.json',
            data_prefix=dict(img=''),
            pipeline=train_pipeline)))
val_dataloader.merge(
    dict(
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/lvis_v1_val.json',
            data_prefix=dict(img=''))))
test_dataloader = val_dataloader

val_evaluator.merge(
    dict(
        type=LVISMetric,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        metric=['bbox', 'segm']))
test_evaluator = val_evaluator

train_cfg.merge(dict(val_interval=24))
