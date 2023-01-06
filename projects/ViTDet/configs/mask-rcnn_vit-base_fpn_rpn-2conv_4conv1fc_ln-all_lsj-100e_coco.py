_base_ = [
    '../../../configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../../../configs/common/lsj-100e_coco-instance.py',
]

custom_imports = dict(imports=['projects.ViTDet.vitdet'])

image_size = (1024, 1024)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]
norm_cfg = dict(type='CustomLN', requires_grad=True)
model = dict(
    # the model is trained from scratch, so init_cfg is None
    data_preprocessor=dict(
        # pad_size_divisor=32 is unnecessary in training but necessary
        # in testing.
        pad_size_divisor=32,
        batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type='VisionTransformer',
        arch='b',
        img_size=1024,
        drop_path_rate=0.1,
        out_indices=(2, 5, 8, 11),
        norm_cfg=norm_cfg,
        init_cfg=None),
    neck=dict(in_channels=[768, 768, 768, 768], norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))
