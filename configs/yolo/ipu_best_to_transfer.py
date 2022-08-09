# copy ipu_best_to_transfer.py from ../../zone2/mmdetection_hudi/configs/myconfigs/best.py
_base_ = '../others/yolo/yolov3_d53_mstrain-608_273e_coco.py'
# dataset settings
IM_SIZE = 320
custom_hooks = [dict(type='NumClassCheckHook'), dict(type='BNToFP32')]
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(IM_SIZE, IM_SIZE), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='UINT8'),
    dict(type='BGR2RGB'),
    dict(type='Pad', size=(IM_SIZE, IM_SIZE)),
    dict(type='IPUFormatBundle',
         img_to_float=False,
         pad_dic=dict(gt_bboxes=dict(dim=0, num=96),
                      gt_labels=dict(dim=0, num=96),
                      gt_bboxes_ignore=dict(dim=0, num=20))),
    dict(type='IPUTempCollect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='GetTargetsOutsideForYolo', featmap_sizes=[IM_SIZE//32, IM_SIZE//16, IM_SIZE//8])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(IM_SIZE, IM_SIZE), keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='UINT8'),
            dict(type='BGR2RGB'),
            dict(type='Pad', size=(IM_SIZE, IM_SIZE), pad_val=dict(img=0.5)),
            dict(type='IPUFormatBundle', img_to_float=False),
            dict(type='IPUTempCollect', keys=['img'], meta_tensor_keys=('scale_factor'), meta_on=True)
        ])
]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(320, 320),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='DefaultFormatBundle'),
#             dict(type='Collect', keys=['img'])
#         ])
# ]

data = dict(
    samples_per_gpu=8*6,
    workers_per_gpu=8,
    train_dataloader=dict(drop_last=True,persistent_workers=True),
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# ipu settings
optimizer_config = dict(_delete_=True)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005, max_grad_norm=35)

options_cfg = dict(
    # randomSeed=42,
    partialsType='half',
    enableExecutableCaching='engine_cache',
    train_cfg=dict(
        executionStrategy='SameAsIpu',
        Training=dict(gradientAccumulation=8),
        availableMemoryProportion=[0.1, 0.1, 0.1, 0.1],
        # accumulationAndReplicationReductionType='Sum',
    ),
    eval_cfg=dict(deviceIterations=1, ),
)

ipu_model_cfg = dict(
    split_edges=[
        dict(layer_to_call='backbone.conv_res_block5', ipu_id=1),
        dict(layer_to_call='neck', ipu_id=2),
        dict(layer_to_call='bbox_head', ipu_id=3),
        # dict(layer_to_call='bbox_head.convs_bridge.0', ipu_id=3),
    ])

# modules_to_record = [
    # 'model/my_catcher_head_out/RemapCE:0',
    # 'model/my_catcher_head_out/RemapCE:0/1',
    # 'model/my_catcher_head_out/RemapCE:0/2',
    # 'model/my_catcher_nms_in/RemapCE:0',
    # 'model/my_catcher_nms_in/RemapCE:0/1',
    # 'model/my_catcher_nms_in/RemapCE:0/2',
    # 'model/my_catcher_nms_out/RemapCE:0',
    # 'model/my_catcher_nms_out/RemapCE:0/1',
    # 'model/my_catcher_nms_out/RemapCE:0/2',
    # 'model/backbone/conv_res_block1/conv/conv/Conv:0',
    # 'model/backbone/conv_res_block3/conv/conv/Conv:0',
    # 'model/backbone/conv_res_block5/conv/bn/BatchNormalization:0',
    # 'model/my_catcher_head_in/RemapCE:0',
    # 'model/my_catcher_head_in/RemapCE:0/1',
    # 'model/my_catcher_head_in/RemapCE:0/2',
    # 'model/neck/detect3/conv1/conv/Conv:0',
    # 'model/neck/detect2/conv1/conv/Conv:0',
    # 'model/neck/detect1/conv1/conv/Conv:0',
    # 'model/neck/Concat:0',
    # 'model/neck/Concat:0/1',
    # 'model/neck/detect1/conv1/conv/Conv:0',
    # 'model/neck/detect1/conv1/bn/BatchNormalization:0',
    # 'model/neck/detect1/conv1/activate/LeakyRelu:0',
    # 'model/neck/my_catcher_neck_in/RemapCE:0',
# ]

runner = dict(type='IPUEpochBasedRunner',
              ipu_model_cfg=ipu_model_cfg,
              options_cfg=options_cfg,
              normalize_im=True,)
            #   modules_to_record=modules_to_record)

# fp16 = dict(loss_scale=256.0, velocity_accum_type='half', accum_type='half')
fp16 = dict(loss_scale=512.0, velocity_accum_type='half')

# original gradient is: 8(batch.sum)*8(replica.mean) = 8 samples' gradient
# ipu use gradient: 8(batch.sum)*8(accumulation.sum)*1(replica.sum)/8 = 8 samples' gradient
model = dict(
    bbox_head=dict(
        # loss_cls=dict(
        #     type='CrossEntropyLoss',
        #     use_sigmoid=True,
        #     loss_weight=1.0/8,
        #     reduction='sum'),
        # loss_conf=dict(
        #     type='CrossEntropyLoss',
        #     use_sigmoid=True,
        #     loss_weight=1.0/8,
        #     reduction='sum'),
        # loss_xy=dict(
        #     type='CrossEntropyLoss',
        #     use_sigmoid=True,
        #     loss_weight=2.0/8,
        #     reduction='sum'),
        # loss_wh=dict(type='MSELoss', loss_weight=2.0/8, reduction='sum'),
        static=False,),
    train_cfg=dict(
        assigner=dict(
            static=False,
        ),
    )
)

# beginblock_idx = 3
serial_num = 6
# stash_layer_names = ['conv_res_block3',]
# only_backbone = True
# block_id_need_to_remap_internally = [3,]
# save_hooked_data = 'data_to_align_for_test.npy'

