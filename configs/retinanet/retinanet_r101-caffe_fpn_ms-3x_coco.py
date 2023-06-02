_base_ = './retinanet_r50-caffe_fpn_ms-3x_coco.py'
# learning policy
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe')))
