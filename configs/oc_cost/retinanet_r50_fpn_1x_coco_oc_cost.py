_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'

data_root = 'data/coco/'

val_evaluator = dict(
    _delete_=True,
    type='CocoOCCostMetric',
    alpha=0.5,
    beta=0.6,
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox')
test_evaluator = val_evaluator
