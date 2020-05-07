_base_ = './mask_rcnn_x101_32x4d_fpn_gn_ws-all_2x_coco.py'
# learning policy
lr_config = dict(step=[20, 23])
total_epochs = 24
