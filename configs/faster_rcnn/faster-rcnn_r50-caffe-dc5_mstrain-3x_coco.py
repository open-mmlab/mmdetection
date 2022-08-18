_base_ = './faster-rcnn_r50-caffe-dc5_mstrain-1x_coco.py'
# learning policy
lr_config = dict(step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)
