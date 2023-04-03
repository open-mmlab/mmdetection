_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_4e_base.py',
    '../_base_/datasets/mot_challenge.py',
]

# evaluator
val_evaluator = [
    dict(type='CocoVideoMetric', metric=['bbox'], classwise=True),
    dict(type='MOTChallengeMetric', metric=['HOTA', 'CLEAR', 'Identity'])
]

test_evaluator = val_evaluator
# The fluctuation of HOTA is about +-1.
randomness = dict(seed=6)
