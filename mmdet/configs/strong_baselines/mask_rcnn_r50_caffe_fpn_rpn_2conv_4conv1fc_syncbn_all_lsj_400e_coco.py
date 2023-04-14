if '_base_':
    from .mask_rcnn_r50_caffe_fpn_rpn_2conv_4conv1fc_syncbn_all_lsj_100e_coco import *
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

# Use RepeatDataset to speed up training
# change repeat time from 4 (for 100 epochs) to 16 (for 400 epochs)
train_dataloader.merge(dict(dataset=dict(times=4 * 4)))
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=0.067,
        by_epoch=False,
        begin=0,
        end=500 * 4),
    dict(
        type=MultiStepLR,
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[22, 24],
        gamma=0.1)
]
