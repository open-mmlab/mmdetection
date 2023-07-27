from mmengine.config import read_base

with read_base():
    from .._base_.datasets.mot_challenge import *
    from .qdtrack_faster_rcnn_r50_fpn_4e_base import *

from mmdet.evaluation import CocoVideoMetric, MOTChallengeMetric

# evaluator
val_evaluator = [
    dict(type=CocoVideoMetric, metric=['bbox'], classwise=True),
    dict(type=MOTChallengeMetric, metric=['HOTA', 'CLEAR', 'Identity'])
]

test_evaluator = val_evaluator
# The fluctuation of HOTA is about +-1.
randomness = dict(seed=6)
