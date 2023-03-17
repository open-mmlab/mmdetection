_base_ = './faster-rcnn_r50_fpn_1x_coco.py'
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768),
               (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
# MMEngine support the following two ways, users can choose
# according to convenience
# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
_base_.train_dataloader.dataset.pipeline = train_pipeline
