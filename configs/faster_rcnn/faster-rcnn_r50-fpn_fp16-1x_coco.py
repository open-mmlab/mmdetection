_base_ = './faster-rcnn_r50-fpn_1x_coco.py'
# fp16 settings
fp16 = dict(loss_scale=512.)
