_base_ = './ms_rcnn_x101_64x4d_fpn_1x_coco.py'
# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
