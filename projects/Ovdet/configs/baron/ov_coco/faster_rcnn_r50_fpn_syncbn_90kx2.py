_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn_syncbn.py',
    '../../_base_/datasets/coco_ovd_base_ms.py',
    '../../_base_/schedules/schedule_90k.py',
    '../../_base_/iter_based_runtime.py'
]
class_weight = [
    1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1,
    0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1,
    1, 0, 0, 0, 1
] + [1]

reg_layer = [
    dict(type='Linear', in_features=1024, out_features=1024),
    dict(type='ReLU', inplace=True),
    dict(type='Linear', in_features=1024, out_features=4)
]

clip_cfg = dict(  # ViT-B/32
    type='CLIP',
    image_encoder=None,
    text_encoder=dict(
        type='CLIPTextEncoder',
        embed_dim=512,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,  # also the word embedding dim
        transformer_heads=8,
        transformer_layers=12,
        init_cfg=dict(
            type='Pretrained', checkpoint='checkpoints/clip_vitb32.pth')))

model = dict(
    type='OVDTwoStageDetector',
    rpn_head=dict(
        type='DetachRPNHead',
        anchor_generator=dict(
            scale_major=False,  # align with detectron2
        )),
    roi_head=dict(
        type='OVDStandardRoIHead',
        clip_cfg=clip_cfg,
        bbox_head=dict(
            type='BaronShared4Conv1FCBBoxHead',
            reg_predictor_cfg=reg_layer,
            reg_class_agnostic=True,
            cls_bias=None,
            num_words=6,
            cls_temp=50.0,
            cls_embeddings_path='data/metadata/' +
            'coco_clip_hand_craft_attn12.npy',
            bg_embedding='learn',
            use_attn12_output=True,
            loss_cls=dict(
                type='CustomCrossEntropyLoss',
                use_sigmoid=False,
                class_weight=class_weight),
        ),
    ),
)

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',  # amp training
    optimizer=dict(
        type='SGD', lr=0.02 * 2, momentum=0.9, weight_decay=0.000025),
    # clip_grad=dict(max_norm=35, norm_type=2),
)
load_from = 'checkpoints/res50_fpn_soco_star_400.pth'
