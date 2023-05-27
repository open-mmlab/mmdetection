_base_ = ['./fcos_r50_caffe_fpn_gn-head_1x_coco.py']

data = dict(samples_per_gpu=8, workers_per_gpu=8)

# optimizer
optimizer = dict(lr=0.04)
fp16 = dict(loss_scale='dynamic')

# learning policy
# In order to avoid non-convergence in the early stage of
# mixed-precision training, the warmup in the lr_config is set to linear,
# warmup_iters increases and warmup_ratio decreases.
lr_config = dict(warmup='linear', warmup_iters=1000, warmup_ratio=1.0 / 10)
