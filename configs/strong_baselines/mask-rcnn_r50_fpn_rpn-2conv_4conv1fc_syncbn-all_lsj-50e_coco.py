_base_ = 'mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-100e_coco.py'

# Use RepeatDataset to speed up training
# change repeat time from 4 (for 100 epochs) to 2 (for 50 epochs)
train_dataloader = dict(dataset=dict(times=2))
