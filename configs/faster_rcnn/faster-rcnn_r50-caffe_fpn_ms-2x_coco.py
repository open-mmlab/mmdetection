_base_ = './faster-rcnn_r50-caffe_fpn_ms-1x_coco.py'

_base_.param_scheduler[1].milestones = [16, 23]
train_cfg = dict(max_epochs=24)
