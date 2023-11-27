_base_ = [
    '../../_base_/models/mask-rcnn_r50_fpn_syncbn.py',
    '../../_base_/datasets/lvis_v1_ovd_kd.py',
    '../../_base_/schedules/schedule_45k.py',
    '../../_base_/iter_based_runtime.py'
]
class_weight = 'data/metadata/lvis_v1_train_cat_norare_info.json'

custom_imports = dict(
    imports=['projects.Ovdet.ovdet'], allow_failed_imports=False)

reg_layer = [
    dict(type='Linear', in_features=1024, out_features=1024),
    dict(type='ReLU', inplace=True),
    dict(type='Linear', in_features=1024, out_features=4)
]

clip_cfg = dict(  # ViT-B/32
    type='CLIP',
    image_encoder=dict(
        type='CLIPViT',
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        init_cfg=dict(
            type='Pretrained',
            prefix='visual',
            checkpoint='checkpoints/clip_vitb32.pth')),
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

ovd_cfg = dict(
    type='BaronKD',
    use_gt=True,
    bag_weight=1.0,
    single_weight=0.1,
    use_attn_mask=False,
    bag_temp=20.0,
    single_temp=30.0,
    clip_data_preprocessor=dict(
        type='ImgDataPreprocessor',
        mean=[(122.7709383 - 123.675) / 58.395, (116.7460125 - 116.28) / 57.12,
              (104.09373615 - 103.53) / 57.375],
        std=[68.5005327 / 58.395, 66.6321579 / 57.12, 70.32316305 / 57.375]),
    num_words=4,
    word_dim=512,
    words_drop_ratio=0.5,
    queue_cfg=dict(
        names=[
            'clip_text_features', 'clip_image_features', 'clip_word_features',
            'clip_patch_features'
        ],
        lengths=[1024] * 4,
        emb_dim=512,
        id_length=1),
    sampling_cfg=dict(
        shape_ratio_thr=0.25,
        area_ratio_thr=0.01,
        objectness_thr=0.85,
        nms_thr=0.1,
        topk=500,
        max_groups=6,
        max_permutations=2,
        alpha=3.0,
        cut_off_thr=0.3,
        base_probability=0.3,
        interval=-0.1,
    ),
)

model = dict(
    type='OVDTwoStageDetector',
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        _delete_=True,
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_mask=True,
            pad_size_divisor=32),
    ),
    rpn_head=dict(
        type='CustomRPNHead',
        anchor_generator=dict(
            scale_major=False,  # align with detectron2
        )),
    batch2ovd=dict(kd_batch='baron_kd'),
    roi_head=dict(
        type='OVDStandardRoIHead',
        clip_cfg=clip_cfg,
        ovd_cfg=dict(baron_kd=ovd_cfg),
        bbox_head=dict(
            type='BaronShared4Conv1FCBBoxHead',
            reg_predictor_cfg=reg_layer,
            reg_class_agnostic=True,
            cls_bias=None,
            cls_temp=100.0,
            test_cls_temp=100 / 0.7,  # follow detpro
            num_classes=1203,
            cls_embeddings_path='projects/Ovdet/' +
            'metadata/lvis_v1_clip_detpro.npy',
            bg_embedding='learn',
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
    clip_grad=dict(max_norm=35, norm_type=2),
)
load_from = 'checkpoints/res50_fpn_soco_star_400.pth'
