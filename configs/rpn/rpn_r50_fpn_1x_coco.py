_base_ = [
    '../_base_/models/rpn_r50_fpn.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

val_evaluator = dict(metric='proposal_fast')
test_evaluator = val_evaluator

# inference on val dataset and dump the proposals with evaluate metric
# data_root = 'data/coco/'
# test_evaluator = [
#     dict(
#         type='DumpProposals',
#         output_dir=data_root + 'proposals/',
#         proposals_file='rpn_r50_fpn_1x_val2017.pkl'),
#     dict(
#         type='CocoMetric',
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         metric='proposal_fast',
#         backend_args={{_base_.backend_args}},
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
#         type='DumpProposals',
#         output_dir=data_root + 'proposals/',
#         proposals_file='rpn_r50_fpn_1x_train2017.pkl'),
# ]
