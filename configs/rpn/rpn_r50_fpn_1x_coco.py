_base_ = [
    '../_base_/models/rpn_r50_fpn.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# TODO: Delete
# val_evaluator = dict(metric='proposal_fast')
val_evaluator = dict(
    _delete_=True,
    type='ProposalRecallMetric',
    proposal_nums=(100, 300, 1000),
    use_legacy_coordinate=False,  # VOCDataset should set True, else False
)
test_evaluator = val_evaluator
