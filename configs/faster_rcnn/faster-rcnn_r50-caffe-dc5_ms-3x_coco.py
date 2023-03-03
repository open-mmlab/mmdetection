_base_ = './faster-rcnn_r50-caffe-dc5_ms-1x_coco.py'

_base_.param_scheduler[1].milestones = [28, 34]
train_cfg = dict(max_epochs=36)
