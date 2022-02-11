_base_ = './retinanet_r50_caffe_fpn_mstrain_1x_coco.py'

model = dict(
    neck=dict(relu_before_extra_convs=True),
    bbox_head=dict(
        loss_normalizer_momentum=0.9,
        loss_normalizer=100,
    ))

data = dict(persistent_workers=True)
