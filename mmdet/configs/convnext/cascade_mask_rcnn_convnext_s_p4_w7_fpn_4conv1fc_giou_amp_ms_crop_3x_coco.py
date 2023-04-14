if '_base_':
    from .cascade_mask_rcnn_convnext_t_p4_w7_fpn_4conv1fc_giou_amp_ms_crop_3x_coco import *

# TODO: delete custom_imports after mmcls supports auto import
# please install mmcls>=1.0
# import mmcls.models to trigger register_module in mmcls
custom_imports.merge(
    dict(imports=['mmcls.models'], allow_failed_imports=False))
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'  # noqa

model.merge(
    dict(
        backbone=dict(
            _delete_=True,
            type='mmcls.ConvNeXt',
            arch='small',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.6,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            init_cfg=dict(
                type='Pretrained',
                checkpoint=checkpoint_file,
                prefix='backbone.'))))

optim_wrapper.merge(
    dict(paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12
    }))
