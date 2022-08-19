_base_ = './mask-rcnn_s50-fpn-syncbn-backbone+head_mstrain-1x_coco.py'
model = dict(
    backbone=dict(
        stem_channels=128,
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='open-mmlab://resnest101')))
