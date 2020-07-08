_base_ = './detr2-1x-1.py'

train_cfg = dict(
    rcnn=dict(
        sampler=dict(
            pos_fraction=0.6))
)