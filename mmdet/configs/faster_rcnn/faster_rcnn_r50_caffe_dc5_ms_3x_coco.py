if '_base_':
    from .faster_rcnn_r50_caffe_dc5_ms_1x_coco import *
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

# MMEngine support the following two ways, users can choose
# according to convenience
# param_scheduler = [
#     dict(
#         type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500), # noqa
#     dict(
#         type=MultiStepLR,
#         begin=0,
#         end=12,
#         by_epoch=True,
#         milestones=[28, 34],
#         gamma=0.1)
# ]
param_scheduler[1].milestones = [28, 34]

train_cfg.merge(dict(max_epochs=36))
