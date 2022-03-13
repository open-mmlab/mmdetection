# TODO: Remove this config after benchmarking all related configs
_base_ = 'fcos_r50_caffe_fpn_gn-head_1x_coco.py'

data = dict(samples_per_gpu=4, workers_per_gpu=4)

# NOTE: This is for automatically scaling LR, USER CAN'T CHANGE THIS VALUE
default_batch_size = 32  # (8 GPUs) x (4 samples per GPU)
