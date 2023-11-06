# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.hooks import CheckpointHook
from mmengine.model.weight_init import PretrainedInit
from mmengine.optim import OptimWrapper
from mmengine.optim.scheduler import MultiStepLR
from mmengine.runner import LogProcessor
from mmengine.runner.loops import IterBasedTrainLoop, TestLoop, ValLoop
from mmengine.visualization import LocalVisBackend
from torch.nn import BatchNorm2d, GroupNorm
from torch.nn.modules.activation import ReLU
from torch.optim.adamw import AdamW

from mmdet.engine.hooks import TrackVisualizationHook
from mmdet.evaluation.metrics.youtube_vis_metric import YouTubeVISMetric
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.data_preprocessors.track_data_preprocessor import \
    TrackDataPreprocessor
from mmdet.models.layers.msdeformattn_pixel_decoder import \
    MSDeformAttnPixelDecoder
from mmdet.models.layers.positional_encoding import SinePositionalEncoding3D
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.dice_loss import DiceLoss
from mmdet.models.task_modules.assigners.hungarian_assigner import \
    HungarianAssigner
from mmdet.models.task_modules.assigners.match_cost import (
    ClassificationCost, CrossEntropyLossCost, DiceCost)
from mmdet.models.task_modules.samplers.mask_pseudo_sampler import \
    MaskPseudoSampler
from mmdet.models.tracking_heads.mask2former_track_head import \
    Mask2FormerTrackHead
from mmdet.models.vis.mask2former_vis import Mask2FormerVideo
from mmdet.visualization import TrackLocalVisualizer

with read_base():
    from .._base_.datasets.youtube_vis import *
    from .._base_.default_runtime import *

num_classes = 40
num_frames = 2
model = dict(
    type=Mask2FormerVideo,
    data_preprocessor=dict(
        type=TrackDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type=BatchNorm2d, requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type=PretrainedInit, checkpoint='torchvision://resnet50')),
    track_head=dict(
        type=Mask2FormerTrackHead,
        in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_frames=num_frames,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type=MSDeformAttnPixelDecoder,
            num_outs=3,
            norm_cfg=dict(type=GroupNorm, num_groups=32),
            act_cfg=dict(type=ReLU),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=128,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type=ReLU, inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type=SinePositionalEncoding3D, num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type=ReLU, inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type=CrossEntropyLoss,
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type=DiceLoss,
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type=HungarianAssigner,
                match_costs=[
                    dict(type=ClassificationCost, weight=2.0),
                    dict(
                        type=CrossEntropyLossCost,
                        weight=5.0,
                        use_sigmoid=True),
                    dict(type=DiceCost, weight=5.0, pred_act=True, eps=1.0)
                ]),
            sampler=dict(type=MaskPseudoSampler))),
    init_cfg=dict(
        type=PretrainedInit,
        checkpoint='https://download.openmmlab.com/mmdetection/v3.0/'
        'mask2former/mask2former_r50_8xb2-lsj-50e_coco/'
        'mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth'))

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.0001, weight_decay=0.05, eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

# learning policy
max_iters = 6000
param_scheduler = dict(
    type=MultiStepLR,
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[
        4000,
    ],
    gamma=0.1)
# runtime settings
train_cfg = dict(
    type=IterBasedTrainLoop, max_iters=max_iters, val_interval=6001)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

vis_backends = [dict(type=LocalVisBackend)]
visualizer = dict(
    type=TrackLocalVisualizer, vis_backends=vis_backends, name='visualizer')

default_hooks.update(
    dict(
        checkpoint=dict(
            type=CheckpointHook, by_epoch=False, save_last=True,
            interval=2000),
        visualization=dict(type=TrackVisualizationHook, draw=False)))

log_processor = dict(type=LogProcessor, window_size=50, by_epoch=False)

# evaluator
val_evaluator = dict(
    type=YouTubeVISMetric,
    metric='youtube_vis_ap',
    outfile_prefix='./youtube_vis_results',
    format_only=True)
test_evaluator = val_evaluator
