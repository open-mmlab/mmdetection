_base_ = ['dino-4scale_r50_8xb2-12e_coco.py']

# from deformable detr hyper
model = dict(
    backbone=dict(frozen_stages=-1),
    bbox_head=dict(loss_cls=dict(loss_weight=2.0)),
    positional_encoding=dict(offset=-0.5, temperature=10000),
    dn_cfg=dict(group_cfg=dict(num_dn_queries=300)))

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.0002),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
