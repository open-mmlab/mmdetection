if '_base_':
    from .faster_rcnn_r50_fpn_1x_coco import *
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmcv.transforms.processing import RandomChoiceResize
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs

model.merge(
    dict(
        data_preprocessor=dict(
            type=DetDataPreprocessor,
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
                checkpoint='open-mmlab://detectron2/resnet50_caffe'))))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=RandomChoiceResize,
        scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768),
               (1333, 800)],
        keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
# MMEngine support the following two ways, users can choose
# according to convenience
# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
train_dataloader.dataset.pipeline = train_pipeline
