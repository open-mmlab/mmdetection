_base_ = './mask_rcnn_r50_fpn_gn-all_contrib_2x_coco.py'

# learning policy
lr_config = dict(step=[28, 34])
total_epochs = 36
