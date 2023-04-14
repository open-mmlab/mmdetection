if '_base_':
    from .._base_.models.rpn_r50_caffe_c4 import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *

val_evaluator.merge(dict(metric='proposal_fast'))
test_evaluator = val_evaluator
