_base_ = 'mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_lsj_100e_coco.py'

# Use RepeatDataset to speed up training
# change repeat time from 4 (for 100 epochs) to 2 (for 50 epochs)
data = dict(train=dict(times=2))
