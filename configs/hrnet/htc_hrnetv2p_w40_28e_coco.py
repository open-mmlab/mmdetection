_base_ = './htc_hrnetv2p_w40_20e_coco.py'
# learning policy
lr_config = dict(step=[24, 27])
total_epochs = 28
