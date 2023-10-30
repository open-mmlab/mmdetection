_base_ = '../glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py'

dataset_type = 'CocoDataset'

# --------------------- Aquarium---------------------#
class_name = ('fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray')
metainfo = dict(classes=class_name)
data_root = '/home/PJLAB/huanghaian/GLIP/odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/'
dataset_Aquarium = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=data_root,
    ann_file='valid/annotations_without_background.json',
    data_prefix=dict(img='valid/'),
    pipeline=_base_.test_pipeline,
    return_classes=True)
val_evaluator_Aquarium = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/annotations_without_background.json',
    metric='bbox')

# --------------------- pothole---------------------#
class_name = ('pothole',)
metainfo = dict(classes=class_name)
data_root = '/home/PJLAB/huanghaian/GLIP/odinw/pothole/'
dataset_pothole = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=data_root,
    ann_file='valid/annotations_without_background.json',
    data_prefix=dict(img='valid/'),
    pipeline=_base_.test_pipeline,
    return_classes=True)
val_evaluator_pothole = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/annotations_without_background.json',
    metric='bbox')


# --------------------- Config---------------------#
dataset_prefixes = ['Aquarium', 'pothole']
datasets = [dataset_Aquarium, dataset_pothole]
metrics = [val_evaluator_Aquarium, val_evaluator_pothole]

# -------------------------------------------------#
val_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=datasets))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator
