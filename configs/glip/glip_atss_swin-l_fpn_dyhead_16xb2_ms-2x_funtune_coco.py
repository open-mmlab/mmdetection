_base_ = './glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco.py'

model = dict(
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        drop_path_rate=0.4,
    ),
    neck=dict(in_channels=[384, 768, 1536]),
    bbox_head=dict(early_fuse=True, num_dyhead_blocks=8, use_checkpoint=True))

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/glip/glip_l_mmdet-abfe026b.pth'  # noqa

optim_wrapper = dict(
    optimizer=dict(lr=0.00001),
    clip_grad=dict(_delete_=True, max_norm=1, norm_type=2))

# TTA
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))

img_scales = [(400, 2500), (500, 2500), (600, 2500), (640, 2500), (700, 2500),
              (900, 2500), (1100, 2500), (1200, 2500), (1400, 2500),
              (1800, 2500)]

tta_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend='pillow'),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='FixScaleResize',
                scale=s,
                keep_ratio=True,
                backend='pillow') for s in img_scales
        ], [
            dict(type='RandomFlip', prob=1.),
            dict(type='RandomFlip', prob=0.)
        ], [dict(type='LoadAnnotations', with_bbox=True)],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'flip',
                                       'flip_direction'))
                    ]])
]
