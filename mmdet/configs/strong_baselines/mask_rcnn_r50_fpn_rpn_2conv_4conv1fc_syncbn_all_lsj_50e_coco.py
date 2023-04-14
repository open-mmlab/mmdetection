if '_base_':
    from .mask_rcnn_r50_fpn_rpn_2conv_4conv1fc_syncbn_all_lsj_100e_coco import *

# Use RepeatDataset to speed up training
# change repeat time from 4 (for 100 epochs) to 2 (for 50 epochs)
train_dataloader.merge(dict(dataset=dict(times=2)))
