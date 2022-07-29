_base_ = ['./dino_4scale_r50_16x2_12e_coco.py']

model = dict(neck=dict(in_channels=[256, 512, 1024, 2048], num_outs=5))
data = dict(samples_per_gpu=1)
