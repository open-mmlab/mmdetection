_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50-caffe-c4.py',
    '../../_base_/datasets/coco_ovd_base.py',
    '../../_base_/schedules/schedule_90k.py',
    '../../_base_/iter_based_runtime.py'
]
class_weight = [
    1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1,
    0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1,
    1, 0, 0, 0, 1
] + [0]

reg_layer = [
    dict(type='Linear', in_features=2048, out_features=2048),
    dict(type='ReLU', inplace=True),
    dict(type='Linear', in_features=2048, out_features=4)
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
        type='CustomRPNHead',
        anchor_generator=dict(
            scale_major=False,  # align with detectron2
        )),
    # backbone=dict(
    #     init_cfg=dict(
    #         checkpoint='checkpoints/resnet50_msra-5891d200.pth')),
    roi_head=dict(
        type='OVDStandardRoIHead',
        # shared_head=dict(
        #     init_cfg=dict(
        #         checkpoint='checkpoints/resnet50_msra-5891d200.pth')),
        clip_cfg=clip_cfg,
        bbox_head=dict(
            type='BaronBBoxHead',
            reg_predictor_cfg=reg_layer,
            reg_class_agnostic=True,
            cls_bias=-20.0,
            cls_temp=25.0,
            cls_embeddings_path='data/metadata/' +
            'coco_clip_hand_craft_attn12.npy',
            use_attn12_output=True,
            loss_cls=dict(
                type='CustomCrossEntropyLoss',
                use_sigmoid=True,
                class_weight=class_weight),
        ),
    ),
)

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',  # amp training
    clip_grad=dict(max_norm=35, norm_type=2),
)
