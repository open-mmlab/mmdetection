if '_base_':
    from .._base_.models.retinanet_r50_fpn import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *

# TODO: delete custom_imports after mmcls supports auto import
# please install mmcls>=1.0
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)

model.merge(
    dict(
        backbone=dict(
            _delete_=True,
            type='mmcls.TIMMBackbone',
            model_name='efficientnet_b1',
            features_only=True,
            pretrained=True,
            out_indices=(1, 2, 3, 4)),
        neck=dict(in_channels=[24, 40, 112, 320])))

# optimizer
optim_wrapper.merge(dict(optimizer=dict(lr=0.01)))
