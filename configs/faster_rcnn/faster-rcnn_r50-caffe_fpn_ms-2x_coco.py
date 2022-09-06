_base_ = './faster-rcnn_r50-caffe_fpn_ms-1x_coco.py'
# learning policy
lr_config = dict(step=[16, 23])
runner = dict(type='EpochBasedRunner', max_epochs=24)
