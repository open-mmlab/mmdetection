_base_ = [
    '../../_base_/models/mask-rcnn_r50_fpn_syncbn.py',
    '../../_base_/datasets/lvis_v1_ovd_base.py',
    '../../_base_/schedules/schedule_45k.py',
    '../../_base_/iter_based_runtime.py'
]
class_weight = 'data/metadata/lvis_v1_train_cat_norare_info.json'

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
        type='CustomRPNHead',
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
            cls_temp=100.0,
            test_cls_temp=100 / 0.7,  # follow detpro
            cls_embeddings_path='data/metadata/lvis_v1_clip_detpro.npy',
            bg_embedding='learn',
            num_classes=1203,
            use_attn12_output=False,
            loss_cls=dict(
                type='CustomCrossEntropyLoss',
                use_sigmoid=False,
                class_weight=class_weight),
        ),
        mask_head=dict(num_classes=1203)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)))

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',  # amp training
    optimizer=dict(
        type='SGD', lr=0.02 * 4, momentum=0.9, weight_decay=0.000025),
    clip_grad=dict(type='value', clip_value=1.0),
)
load_from = 'checkpoints/res50_fpn_soco_star_400.pth'
