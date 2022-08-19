_base_ = 'mask-rcnn_r50-fpn-syncbn-all-rpn-2conv_lsj-100e_coco.py'

# Use RepeatDataset to speed up training
# change repeat time from 4 (for 100 epochs) to 2 (for 50 epochs)
train_dataloader = dict(dataset=dict(times=2))
