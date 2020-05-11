_base_ = './fcos_hrnetv2p_w32_gn-head_4x4_1x_coco.py'
# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24
