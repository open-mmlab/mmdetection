_base_ = './mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco.py'

# Use RepeatDataset to speed up training
# change repeat time from 4 (for 100 epochs) to 16 (for 400 epochs)
train_dataloader = dict(dataset=dict(times=4 * 4))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.067,
        by_epoch=False,
        begin=0,
        end=500 * 4),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[22, 24],
        gamma=0.1)
]
