if '_base_':
    from .soft_teacher_faster_rcnn_r50_caffe_fpn_180k_semi_0_1_coco import *

# 5% coco train2017 is set as labeled dataset
labeled_dataset = labeled_dataset
unlabeled_dataset = unlabeled_dataset
labeled_dataset.ann_file = 'semi_anns/instances_train2017.1@5.json'
unlabeled_dataset.ann_file = 'semi_anns/instances_train2017.1@5-unlabeled.json'
train_dataloader.merge(
    dict(dataset=dict(datasets=[labeled_dataset, unlabeled_dataset])))
