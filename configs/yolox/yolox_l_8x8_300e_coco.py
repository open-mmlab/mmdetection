_base_ = './yolox_s_8x8_300e_coco.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256))

# NOTE: This is for automatically scaling LR, USER CAN'T CHANGE THIS VALUE
default_batch_size = 64  # (8 GPUs) x (8 samples per GPU)
