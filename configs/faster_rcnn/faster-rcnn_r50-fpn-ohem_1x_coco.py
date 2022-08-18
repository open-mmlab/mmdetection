_base_ = './faster-rcnn_r50-fpn_1x_coco.py'
model = dict(train_cfg=dict(rcnn=dict(sampler=dict(type='OHEMSampler'))))
