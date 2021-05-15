_base_ = './mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
# learning policy
lr_config = dict(step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)
