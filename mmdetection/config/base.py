# fp16 settings
#fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='CascadeRCNN',
    num_stages=3,
    pretrained='torchvision://resnet50',
	backbone=dict(
		type='ResNet',
		depth=50,
		num_stages=4,
		out_indices=(0, 1, 2, 3),
		frozen_stages=1,
		style='pytorch',
        # 在最后三个block加入可变形卷积
		dcn=dict(modulated=False, deformable_groups=1, fallback_on_stride=False),
		        stage_with_dcn=(False, True, True, True)),
	neck=dict(
		type='FPN',
		in_channels=[256, 512, 1024, 2048],
		out_channels=256,
		num_outs=5),
	rpn_head=dict(
		type='RPNHead',
		in_channels=256,
		feat_channels=256,
		anchor_scales=[8],
		anchor_ratios=[0.2, 0.5, 1.0, 2.0, 5.0], #可根据尺寸分布，修改anchor的长宽比
		anchor_strides=[4, 8, 16, 32, 64],
		target_means=[.0, .0, .0, .0],
		target_stds=[1.0, 1.0, 1.0, 1.0],
		# loss_cls=dict(type='CrossEntropyLoss',use_sigmoid=True, loss_weight=1.0),#此处可替换成focalloss
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, loss_weight=1.0),
		loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
	bbox_roi_extractor=dict(
		type='SingleRoIExtractor',
		roi_layer=dict(
			type='RoIAlign',
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
		num_classes=11,#类别数+1(背景类)
		target_means=[0., 0., 0., 0.],
		target_stds=[0.1, 0.1, 0.2, 0.2],
		reg_class_agnostic=False, # 可以试一试True
		loss_cls=dict(
			type='CrossEntropyLoss',
			use_sigmoid=False,
			loss_weight=1.0),
		loss_bbox=dict(
			type='SmoothL1Loss',
			beta=1.0,
			loss_weight=1.0))
)

# model training and testing settings
train_cfg = dict(
	rpn=dict(
		assigner=dict(
			type='MaxIoUAssigner',
			pos_iou_thr=0.7,
			neg_iou_thr=0.3,
			min_pos_iou=0.3,
			ignore_iof_thr=-1),
		sampler=dict(
			type='RandomSampler',#默认使用的是随机采样RandomSampler，这里可替换成OHEMSampler，引入在线难样本学习
			num=256,
			pos_fraction=0.5,
			neg_pos_ub=-1,
			add_gt_as_proposals=False
        ),
		allowed_border=0,
		pos_weight=-1,
		debug=False),
	rpn_proposal=dict(
		nms_across_levels=False,
		nms_pre=2000,
		nms_post=2000,
		max_num=2000,
		nms_thr=0.7,
		min_bbox_size=0),
	rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.4,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler', #默认使用的是随机采样RandomSampler，这里可替换成OHEMSampler，引入在线难样本学习
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weights=[1, 0.5, 0.25]
)

test_cfg = dict(
	rpn=dict(
		nms_across_levels=False,
		nms_pre=1000,
		nms_post=1000,
		max_num=1000,
		nms_thr=0.7,
		min_bbox_size=0),
	rcnn=dict(
		score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
		# soft-nms is also supported for rcnn testing
		# e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'Tianchi1Dataset' #自定义dataloder
data_root = 'data/tianchi1/' #数据集根目录
img_norm_cfg = dict(
	mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadAnnotations', with_bbox=True),
	dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),#修改图像尺寸为半图，可修改为全图训练
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='Pad', size_divisor=32),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']), ]
test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(
		type='MultiScaleFlipAug',
		img_scale=[(1333, 800)],
		flip=False,
		transforms=[
			dict(type='Resize', keep_ratio=True),
			dict(type='RandomFlip'),
			dict(type='Normalize', **img_norm_cfg),
			dict(type='Pad', size_divisor=32),
			dict(type='ImageToTensor', keys=['img']),
			dict(type='Collect', keys=['img']), ]) ]
data = dict(
	imgs_per_gpu=4,
	workers_per_gpu=2,
	train=dict(
		type=dataset_type,
		ann_file=data_root + 'chongqing1_round1_train1_20191223/annotations_washed_train.json', #修改成自己的训练集标注文件路径
		img_prefix=data_root + 'chongqing1_round1_train1_20191223/images/', #训练图片路径
        pipeline=train_pipeline),
	val=dict(
		type=dataset_type,
		ann_file=data_root + 'chongqing1_round1_train1_20191223/annotations_washed_val.json', #修改成自己的验证集标注文件路径
		img_prefix=data_root + 'chongqing1_round1_train1_20191223/images/', #验证图片路径
        pipeline=test_pipeline),
	test=dict(
		type=dataset_type,
		ann_file=data_root + 'chongqing1_round1_train1_20191223/annotations_washed_val.json',#修改成自己的验证集标注文件路径
		img_prefix=data_root + 'chongqing1_round1_train1_20191223/images/', # 'chongqing1_round1_testA_20191223/images/', #测试图片路径
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
	type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001) #学习率的设置尤为关键：lr = 0.00125*batch_size
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
	policy='step',
	warmup='linear',
	warmup_iters=500,
	warmup_ratio=1.0 / 3,
	step=[8, 14, 16, ])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
	interval=50,
	hooks=[ dict(type='TextLoggerHook'),
	# dict(type='TensorboardLoggerHook')
    ]
            )
# yapf:enable
# runtime settings
total_epochs = 18
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/base' #训练的权重和日志保存路径
load_from = None #'./work_dirs/base' #采用coco预训练模型 ,需要对权重类别数进行处理
resume_from = None
workflow = [('train', 1)]