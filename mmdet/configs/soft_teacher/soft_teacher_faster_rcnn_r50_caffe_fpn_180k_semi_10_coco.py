# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.datasets.semi_coco_detection import *
    from .._base_.default_runtime import *

from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import IterBasedTrainLoop, TestLoop
from torch.optim.sgd import SGD

from mmdet.engine.hooks.mean_teacher_hook import MeanTeacherHook
from mmdet.engine.runner import TeacherStudentValLoop
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.data_preprocessors.data_preprocessor import (
    DetDataPreprocessor, MultiBranchDataPreprocessor)
from mmdet.models.detectors.soft_teacher import SoftTeacher

detector = model
detector.data_preprocessor.update(
    dict(
        type=DetDataPreprocessor,
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32))

model = dict(
    type=SoftTeacher,
    detector=detector,
    data_preprocessor=dict(
        type=MultiBranchDataPreprocessor,
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=4.0,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_thr=0.9,
        cls_pseudo_thr=0.9,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

# 10% coco train2017 is set as labeled dataset
labeled_dataset = labeled_dataset
unlabeled_dataset = unlabeled_dataset
labeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10.json'
unlabeled_dataset.ann_file = 'semi_anns/' \
                             'instances_train2017.1@10-unlabeled.json'
unlabeled_dataset.data_prefix = dict(img='train2017/')
train_dataloader.update(
    dict(dataset=dict(datasets=[labeled_dataset, unlabeled_dataset])))

# training schedule for 180k
train_cfg = dict(type=IterBasedTrainLoop, max_iters=180000, val_interval=5000)
val_cfg = dict(type=TeacherStudentValLoop)
test_cfg = dict(type=TestLoop)

# learning rate policy
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=180000,
        by_epoch=False,
        milestones=[120000, 160000],
        gamma=0.1)
]

# optimizer
# The learning rate in the old configuration was 0.01,
# but there was a loss cls error during runtime.
# Because the process of calculating losses is too close to zero,
# the learning rate is adjusted to 0.001.
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=SGD, lr=0.001, momentum=0.9, weight_decay=0.0001))

default_hooks.update(
    dict(checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=2)))
log_processor.update(dict(by_epoch=False))

custom_hooks = [dict(type=MeanTeacherHook)]
