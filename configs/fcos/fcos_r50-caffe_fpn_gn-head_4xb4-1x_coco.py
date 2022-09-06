# TODO: Remove this config after benchmarking all related configs
_base_ = 'fcos_r50-caffe_fpn_gn-head_1x_coco.py'

# dataset settings
train_dataloader = dict(batch_size=4, num_workers=4)
