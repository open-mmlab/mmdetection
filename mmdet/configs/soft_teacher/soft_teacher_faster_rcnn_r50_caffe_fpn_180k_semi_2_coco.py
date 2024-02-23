# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .soft_teacher_faster_rcnn_r50_caffe_fpn_180k_semi_10_coco import *

# 1% coco train2017 is set as labeled dataset
labeled_dataset = labeled_dataset
unlabeled_dataset = unlabeled_dataset
labeled_dataset.ann_file = 'semi_anns/instances_train2017.1@2.json'
unlabeled_dataset.ann_file = 'semi_anns/instances_train2017.1@2-unlabeled.json'
train_dataloader.update(
    dict(dataset=dict(datasets=[labeled_dataset, unlabeled_dataset])))
