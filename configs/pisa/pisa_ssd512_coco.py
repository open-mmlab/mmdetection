_base_ = '../ssd/ssd512_coco.py'

model = dict(
    bbox_head=dict(type='PISASSDHead'),
    train_cfg=dict(isr=dict(k=2., bias=0.), carl=dict(k=1., bias=0.2)))

default_hooks = dict(
    optimizer=dict(
        _delete_=True,
        type='OptimizerHook',
        grad_clip=dict(max_norm=35, norm_type=2)))
