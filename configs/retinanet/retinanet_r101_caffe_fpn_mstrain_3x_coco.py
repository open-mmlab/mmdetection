_base_ = './retinanet_r50_caffe_fpn_mstrain_1x_coco.py'
# learning policy
model = dict(
    backbone=dict(depth=101), init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://detectron2/resnet101_caffe'))
lr_config = dict(step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)
