_base_ = './retinanet_r50_caffe_fpn_mstrain_1x_coco.py'

model = dict(
    bbox_head=dict(
        loss_normalizer_momentum=0.9,
        loss_normalizer=100,
    ))

data = dict(persistent_workers=True)
