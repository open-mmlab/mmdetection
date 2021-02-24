_base_ = './mask_rcnn_r50_fpn_gn-all_2x_coco.py'

# learning policy
lr_config = dict(step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)
