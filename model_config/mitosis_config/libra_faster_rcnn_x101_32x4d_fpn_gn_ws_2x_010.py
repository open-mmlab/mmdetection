# fp16 settings
# fp16 = dict(loss_scale=512.)

# model settings
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='FasterRCNN',
    pretrained='open-mmlab://jhu/resnext101_32x4d_gn_ws',
    backbone=dict(type='ResNeXt',
                  depth=101,
                  groups=32,
                  base_width=4,
                  num_stages=4,
                  out_indices=(0, 1, 2, 3),
                  frozen_stages=1,
                  style='pytorch',
                  conv_cfg=conv_cfg,
                  norm_cfg=norm_cfg),
    neck=[
        dict(type='FPN',
             in_channels=[256, 512, 1024, 2048],
             out_channels=256,
             num_outs=5,
             conv_cfg=conv_cfg,
             norm_cfg=norm_cfg),
        dict(type='BFP',
             in_channels=256,
             num_levels=5,
             refine_level=2,
             refine_type='non_local',
             conv_cfg=conv_cfg,
             norm_cfg=norm_cfg)
    ],
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[4],  # modified by Lichao Wang
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(type='CrossEntropyLoss',
                      use_sigmoid=True,
                      loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(type='SingleRoIExtractor',
                            roi_layer=dict(type='RoIAlign',
                                           out_size=7,
                                           sample_num=2),
                            out_channels=256,
                            featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2,  # modified by Lichao Wang
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        loss_cls=dict(type='CrossEntropyLoss',
                      use_sigmoid=False,
                      loss_weight=1.0),
        loss_bbox=dict(type='BalancedL1Loss',
                       alpha=0.5,
                       gamma=1.5,
                       beta=1.0,
                       loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(assigner=dict(type='MaxIoUAssigner',
                           pos_iou_thr=0.7,
                           neg_iou_thr=0.3,
                           min_pos_iou=0.3,
                           ignore_iof_thr=-1),
             sampler=dict(type='RandomSampler',
                          num=256,
                          pos_fraction=0.5,
                          neg_pos_ub=5,
                          add_gt_as_proposals=False),
             allowed_border=-1,
             pos_weight=-1,
             debug=False),
    rpn_proposal=dict(nms_across_levels=False,
                      nms_pre=2000,
                      nms_post=2000,
                      max_num=2000,
                      nms_thr=0.7,
                      min_bbox_size=0),
    rcnn=dict(
        assigner=dict(type='MaxIoUAssigner',
                      pos_iou_thr=0.5,
                      neg_iou_thr=0.5,
                      min_pos_iou=0.5,
                      ignore_iof_thr=-1),
        sampler=dict(type='OHEMSampler',
                     num=512,
                     pos_fraction=0.25,
                     neg_pos_ub=-1,
                     add_gt_as_proposals=True),
        #   sampler=dict(type='CombinedSampler',
        #                num=512,
        #                pos_fraction=0.25,
        #                add_gt_as_proposals=True,
        #                pos_sampler=dict(type='InstanceBalancedPosSampler'),
        #                neg_sampler=dict(type='IoUBalancedNegSampler',
        #                                 floor_thr=-1,
        #                                 floor_fraction=0,
        #                                 num_bins=3)),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(nms_across_levels=False,
             nms_pre=1000,
             nms_post=1000,
             max_num=1000,
             nms_thr=0.7,
             min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.1),  # 0.5),  # modified by Lichao Wang
        max_per_img=20)  # 100)  # modified by Lichao Wang
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'MitosisDataset'  # 'VOCDataset'
data_root = 'data/VOC_mitos/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
albu_train_transforms = [
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5),
    dict(type='RandomRotate90', p=0.5),
    dict(type='ShiftScaleRotate',
         shift_limit=0.0,
         scale_limit=0.05,
         rotate_limit=30,
         interpolation=1,
         p=0.25),
    # dict(type='RandomBrightnessContrast',
    #      brightness_limit=[0.1, 0.3],
    #      contrast_limit=[0.1, 0.3],
    #      p=0.2),
    # dict(type='OneOf',
    #      transforms=[
    #          dict(type='RGBShift', r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
    #          dict(type='HueSaturationValue', hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0)
    #      ],
    #      p=0.1),
    # dict(type='OneOf',
    #      transforms=[
    #          dict(type='Blur', blur_limit=3, p=1.0),
    #          dict(type='MedianBlur', blur_limit=3, p=1.0)
    #      ],
    #      p=0.1),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='Albu',
         transforms=albu_train_transforms,
         bbox_params=dict(type='BboxParams',
                          format='pascal_voc',
                          label_fields=['gt_labels'],   # category_id
                          min_visibility=0.5,  # modified by Lichao Wang
                        #   min_area=0.0,  # added by Lichao Wang
                          filter_lost_elements=True),
         keymap={
             'img': 'image',
             'gt_masks': 'masks',
             'gt_bboxes': 'bboxes'
         },
         update_pad_shape=False,
         skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg', 'pad_shape', 'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1024, 1024),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]

data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',  # to avoid reloading datasets frequently
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007_OURS_r1024_ALL/ImageSets/Main/positive_train_20200107.txt',
                data_root + 'VOC2007_OURS_r1024_694221/ImageSets/Main/trainval_positive.txt',
                data_root + 'VOC2007_OURS_r1024_728336/ImageSets/Main/trainval_positive.txt',
                data_root + 'VOC2007_OURS_r1024_744729/ImageSets/Main/trainval_positive.txt',
                data_root + 'VOC2007_OURS_r1024_912941/ImageSets/Main/trainval_positive.txt',
                data_root +'VOC2007_OURS_r1024_ICPR/ImageSets/Main/trainval_positive.txt',
            ],
            img_prefix=[
                data_root + 'VOC2007_OURS_r1024_ALL/',
                data_root + 'VOC2007_OURS_r1024_694221/',
                data_root + 'VOC2007_OURS_r1024_728336/',
                data_root + 'VOC2007_OURS_r1024_744729/',
                data_root + 'VOC2007_OURS_r1024_912941/',
                data_root + 'VOC2007_OURS_r1024_ICPR/',
            ],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007_OURS_r1024_ALL/ImageSets/Main/positive_test_20200107.txt',
        img_prefix=data_root + 'VOC2007_OURS_r1024_ALL/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007_OURS_r1024_ALL/ImageSets/Main/positive_test_20200107.txt',
        img_prefix=data_root + 'VOC2007_OURS_r1024_ALL/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1.0 / 3,
                 step=[8, 16, 22])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=1)
# runtime settings
total_epochs = 25  # actual epoch = 4 * 3 = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/libra_faster_rcnn_x101_32x4d_fpn_gn_ws_2x_010'
load_from = None
resume_from = None
workflow = [('train', 1)]
