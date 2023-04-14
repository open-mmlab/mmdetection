if '_base_':
    from ..common.ms_3x_coco_instance import *
    from .._base_.models.cascade_mask_rcnn_r50_fpn import *

model.merge(
    dict(
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
                checkpoint='open-mmlab://detectron2/resnet50_caffe'))))
