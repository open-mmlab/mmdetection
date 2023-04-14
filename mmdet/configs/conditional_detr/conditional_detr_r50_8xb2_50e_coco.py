if '_base_':
    from ..detr.detr_r50_8xb2_150e_coco import *
from mmdet.models.detectors.conditional_detr import ConditionalDETR
from mmdet.models.dense_heads.conditional_detr_head import ConditionalDETRHead
from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.models.task_modules.assigners.hungarian_assigner import HungarianAssigner
from mmdet.models.task_modules.assigners.match_cost import FocalLossCost, BBoxL1Cost, IoUCost
from mmengine.runner.loops import EpochBasedTrainLoop
from mmengine.optim.scheduler.lr_scheduler import MultiStepLR

model.merge(
    dict(
        type=ConditionalDETR,
        num_queries=300,
        decoder=dict(
            num_layers=6,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    _delete_=True,
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1,
                    cross_attn=False),
                cross_attn_cfg=dict(
                    _delete_=True,
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1,
                    cross_attn=True))),
        bbox_head=dict(
            type=ConditionalDETRHead,
            loss_cls=dict(
                _delete_=True,
                type=FocalLoss,
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0)),
        # training and testing settings
        train_cfg=dict(
            assigner=dict(
                type=HungarianAssigner,
                match_costs=[
                    dict(type=FocalLossCost, weight=2.0),
                    dict(type=BBoxL1Cost, weight=5.0, box_format='xywh'),
                    dict(type=IoUCost, iou_mode='giou', weight=2.0)
                ]))))

# learning policy
train_cfg.merge(dict(type=EpochBasedTrainLoop, max_epochs=50, val_interval=1))

param_scheduler = [dict(type=MultiStepLR, end=50, milestones=[40])]
