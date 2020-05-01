_base_ = './mask_rcnn_hrnetv2p_w32_1x_coco.py'
# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24
