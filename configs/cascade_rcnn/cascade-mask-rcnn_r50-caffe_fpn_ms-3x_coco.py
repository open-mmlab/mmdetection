_base_ = [
    '../common/ms_3x_coco-instance.py',
    '../_base_/models/cascade-mask-rcnn_r50_fpn.py'
]

model = dict(
    # use caffe img_norm
    data_preprocessor=dict(
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False),
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))
