_base_ = './retinanet_r50_fpn_1x_coco_v1.py'
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        # use caffe img_norm
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')))
