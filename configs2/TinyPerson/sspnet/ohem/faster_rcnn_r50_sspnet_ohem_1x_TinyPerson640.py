_base_ = [
    '../base/faster_rcnn_r50_sspnet_1x_TinyPerson640.py',
]
train_cfg=dict(rcnn=dict(sampler=dict(type='OHEMSampler')))

