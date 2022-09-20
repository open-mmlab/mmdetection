_base_ = '../yolox/yolox_s_8x8_300e_coco.py'
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.RepVGG',
        arch='small',
        add_ppf=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        out_indices=(1, 2, 3),
    ),
    neck=dict(act_cfg=dict(type='SiLU')),
    bbox_head=dict(act_cfg=dict(type='SiLU')))
