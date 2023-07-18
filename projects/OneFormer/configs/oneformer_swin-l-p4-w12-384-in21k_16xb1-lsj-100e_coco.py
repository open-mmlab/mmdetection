_base_ = ['./oneformer_swin-t-p4-w7-224_lsj_8x2_50e_coco_panoptic.py']
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=192,
        depths=depths,
        num_heads=[6, 12, 24, 48],
        window_size=12,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        ),
    panoptic_head=dict(
        num_queries=150, 
        in_channels=[192, 384, 768, 1536],
        task='instance'
        ),
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,
        ),
    )

data_root='data/coco/'
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/instances_val2017.json',
        metric=['bbox', 'segm'],
        backend_args={{_base_.backend_args}})
]
test_evaluator = val_evaluator
