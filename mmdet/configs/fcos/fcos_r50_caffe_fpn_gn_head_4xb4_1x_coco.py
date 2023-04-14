if '_base_':
    from .fcos_r50_caffe_fpn_gn_head_1x_coco import *
# TODO: Remove this config after benchmarking all related configs

# dataset settings
train_dataloader.merge(dict(batch_size=4, num_workers=4))
