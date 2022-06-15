_base_ = '../ssd/ssd300_coco.py'

model = dict(
    bbox_head=dict(type='PISASSDHead'),
    train_cfg=dict(isr=dict(k=2., bias=0.), carl=dict(k=1., bias=0.2)))

optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))
