# model settings
norm_cfg = dict(type='BN', requires_grad=False)
model = dict(
	type='RetinaNet',
	backbone=dict(
		type='EfficientNet',
		model_name='tf_efficientnet_b2'),
	neck=dict(
		type='BIFPN',
		in_channels=[48, 88, 120, 208, 352], # [JW] 이 정보는 efficientnet에서..?
		out_channels=112,
		start_level=0,
		stack=4, # [JW] BiFPN layers가 d2에선 5, d4에선 7 인데, config에선 stack 값을 각각 4, 6으로 주고 있음
		add_extra_convs=True,
		num_outs=5,
		norm_cfg=norm_cfg,
		activation='relu'),
	bbox_head=dict(
		type='RetinaHead',
		num_classes=81,
		in_channels=112, # 256->112
		stacked_convs=3, # 4->3
		feat_channels=112, # 256->112
		octave_base_sclae=4,# [JW] ??
		scales_per_octave=3, # [JW] ??
		anchor_ratios=[0.5, 1.0, 2.0],
		anchor_strides=[8, 16, 32, 64, 128],
		target_means=[.0, .0, .0, .0], # [JW] ??
		target_stds=[1.0, 1.0, 1.0, 1.0], # [JW] ??
		loss_cls=dict(
			type='FocalLoss',
			use_sigmoid=True,
			gamma=1.5, # 2->1.5
			alpha=0.25,
			loss_weight=1.0),
		loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))

# training and testing settings
train_cfg = dict(
	assigner=dict(
		type='MaxIoUAssigner',
		pos_iou_thr=0.5,
		neg_iou_thr=0.4,
		min_pos_iou=0, # [JW] ??
		ignore_iof_thr=-1), # [JW] ??
	allowed_border=-1, # [JW] ??
	pos_weight=-1, # [JW] ??
	debug=False) # [JW] ?? 
test_cfg = dict(
	nms_pre=1000,
	min_box_size=0,
	score_thr=0.05,
	nms=dict(type='nms', iou_thr=0.5),
	max_per_img=100)

#dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
	mean=[123.675, 116.28, 103.53], 
	std=[58.395, 57.12, 57.375], 
	to_rgb=True)
train_pipeline = [
	dict(type='LodaImageFromFile'),
	dict(type='LoadAnnotations', with_bbox=True),
	dict(tpye='Resize', img_scale=(768, 768), keep_ratio=True),
	dict(tpye='RandomFlip', flip_ratio=0.5),
	dict(tpye='Normalize', **img_norm_cfg),
	dict(tpye='Pad', size_divisor=128), # [JW] ??
	dict(tpye='DefaultFormatBundle'),
	dict(tpye='Collect', keys=['img', 'gt_bboxed', 'gt_labels']),
]
test_pipeline =[
	dict(tpye='LodaImageFromFile'),
	dict(
		type='MultiScalarFlipAug',
		img_scale=(768, 768),
		flip=False,
		transforms=[
			dict(tpye='Resize', keep_ratio=True),
			dict(tpye='RandomFlip'),
			dict(tpye='Normalize', **img_norm_cfg),
			dict(tpye='Pad', size_divisor=128),
			dict(tpye='ImageToTensor', keys=['img']),
			dict(tpye='Collect', keys=['img']),
		])
]
data = dict(
	imgs_per_gpu=2,
	workers_per_gpu=2,
	train=dict(
		type=dataset_type,
		ann_file=data_root + 'annotations/instances_train2017.json',
		img_prefix=data_root + 'train2017/',
		pipeline=train_pipeline),
	val=dict(
		tpye=dataset_type,
		ann_file=data_root + 'annotations/instances_val2017.json',
		img_prefix=data_root + 'val2017/',
		pipeline=test_pipeline),
	test=dict(
		tpye=dataset_type,
		ann_file=data_root + 'annotations/instances_val2017.json',
		img_prefix=data_root + 'val2017/',
		pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum-0.9, weight_decay=4e-5) # wd 0.0001->4e-5
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2)) # [JW] ??
# learning policy
lr_config = dict(
	policy='step',
	warmup='linear',
	warmup_iters=500,
	warmup_ratop=1.0 / 3,
	step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable => [JW] yapf ??
log_config = dict(
	interval=50,
	hooks=[
	dict(type='TextLoggerHook'),
	# dict(type='TensorboardLoggerHook')
	])
# yapf:enable

# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
wordk_dir = './work_dirs/efficient_d2_bifpn_1x' # [JW] 어떻게 설정?
load_from = None
resume_from = None
workflow = [('train', 1)]


		
	













