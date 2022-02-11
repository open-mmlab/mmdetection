_base_ = './retinanet_r50_caffe_fpn_mstrain_loss-ema-norm_1x_coco.py'

model = dict(
    neck=dict(
        relu_before_extra_convs=True,
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d')))
lr_config = dict(warmup_iters=1000)
