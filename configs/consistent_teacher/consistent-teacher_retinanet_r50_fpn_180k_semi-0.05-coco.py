_base_ = ['./consistent-teacher_retinanet_r50_fpn_180k_semi-0.1-coco.py']

# 5% coco train2017 is set as labeled dataset
fold = 1
percent = 5
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = 'semi_anns/instances_train2017.${fold}@${percent}.json'
unlabeled_dataset.ann_file = 'semi_anns/' \
                             'instances_train2017.${fold}@${percent}-unlabeled.json'
