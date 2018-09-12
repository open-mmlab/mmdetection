# model settings
model = dict(
    pretrained=
    '/mnt/lustre/pangjiangmiao/initmodel/pytorch/resnet50-19c8e357.pth',
    backbone=dict(
        type='resnet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='fb'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        coarsest_stride=32,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        use_sigmoid_cls=True),
    roi_block=dict(
        type='SingleLevelRoI',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCRoIHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=81,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False))
meta_params = dict(
    rpn_train_cfg = dict(
        pos_fraction=0.5,
        pos_balance_sampling=False,
        neg_pos_ub=256,
        allowed_border=0,
        anchor_batch_size=256,
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        neg_balance_thr=0,
        min_pos_iou=1e-3,
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False),
    rpn_test_cfg = dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn_train_cfg = dict(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        crowd_thr=1.1,
        roi_batch_size=512,
        add_gt_as_proposals=True,
        pos_fraction=0.25,
        pos_balance_sampling=False,
        neg_pos_ub=512,
        neg_balance_thr=0,
        pos_weight=-1,
        debug=False),
    rcnn_test_cfg = dict(score_thr=1e-3, max_per_img=100, nms_thr=0.5)
)
# dataset settings
data_root = '/mnt/lustre/pangjiangmiao/dataset/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
img_per_gpu = 1
data_workers = 2
train_dataset = dict(
    ann_file=data_root + 'annotations/instances_train2017.json',
    img_prefix=data_root + 'train2017/',
    img_scale=(1333, 800),
    img_norm_cfg=img_norm_cfg,
    size_divisor=32,
    flip_ratio=0.5)
test_dataset = dict(
    ann_file=data_root + 'annotations/instances_val2017.json',
    img_prefix=data_root + 'val2017/',
    img_scale=(1333, 800),
    img_norm_cfg=img_norm_cfg,
    size_divisor=32)
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
grad_clip_config = dict(grad_clip=True, max_norm=35, norm_type=2)
# learning policy
lr_policy = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.333,
    step=[8, 11])
max_epoch = 12
checkpoint_config = dict(interval=1)
dist_params = dict(backend='nccl', port='29500', master_ip='127.0.0.1')
# logging settings
log_level = 'INFO'
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # ('TensorboardLoggerHook', dict(log_dir=work_dir + '/log')),
    ])
# yapf:enable
work_dir = './model/r50_fpn_frcnn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
