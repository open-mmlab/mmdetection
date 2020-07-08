_base_ = './detr2-1x-1.py'

train_cfg = dict(
    rpn_proposal=dict(
        nms_post=600,
        max_num=600)
)