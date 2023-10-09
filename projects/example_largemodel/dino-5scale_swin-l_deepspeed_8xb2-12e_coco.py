from mmengine.config import read_base

with read_base():
    from mmdet.configs.dino.dino_5scale_swin_l_8xb2_12e_coco import *  # noqa

model.update(dict(encoder=dict(num_cp=6)))  # noqa

runner_type = 'FlexibleRunner'
strategy = dict(
    type='DeepSpeedStrategy',
    gradient_clipping=0.1,
    fp16=dict(
        enabled=True,
        fp16_master_weights_and_grads=False,
        loss_scale=0,
        loss_scale_window=500,
        hysteresis=2,
        min_loss_scale=1,
        initial_scale_power=15,
    ),
    inputs_to_half=['inputs'],
    zero_optimization=dict(
        stage=3,
        allgather_partitions=True,
        reduce_scatter=True,
        allgather_bucket_size=50000000,
        reduce_bucket_size=50000000,
        overlap_comm=True,
        contiguous_gradients=True,
        cpu_offload=False),
)

optim_wrapper = dict(
    type='DeepSpeedOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    # clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

# To debug
default_hooks.update(dict(logger=dict(interval=1)))  # noqa
log_processor.update(dict(window_size=1))  # noqa
