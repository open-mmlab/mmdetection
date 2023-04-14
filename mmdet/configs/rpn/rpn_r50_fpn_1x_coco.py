if '_base_':
    from .._base_.models.rpn_r50_fpn import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmdet.evaluation.metrics.dump_proposals_metric import DumpProposals, DumpProposals
from mmdet.evaluation.metrics.coco_metric import CocoMetric

val_evaluator.merge(dict(metric='proposal_fast'))
test_evaluator = val_evaluator

# inference on val dataset and dump the proposals with evaluate metric
# data_root = 'data/coco/'
# test_evaluator = [
#     dict(
#         type=DumpProposals,
#         output_dir=data_root + 'proposals/',
#         proposals_file='rpn_r50_fpn_1x_val2017.pkl'),
#     dict(
#         type=CocoMetric,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         metric='proposal_fast',
#         backend_args=backend_args,
#         format_only=False)
# ]

# inference on training dataset and dump the proposals without evaluate metric
# data_root = 'data/coco/'
# test_dataloader = dict(
#     dataset=dict(
#         ann_file='annotations/instances_train2017.json',
#         data_prefix=dict(img='train2017/')))
#
# test_evaluator = [
#     dict(
#         type=DumpProposals,
#         output_dir=data_root + 'proposals/',
#         proposals_file='rpn_r50_fpn_1x_train2017.pkl'),
# ]
