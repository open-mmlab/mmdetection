_base_ = 'xdecoder-tiny_zeroshot_open-vocab-semseg.py'

model = dict(task='instance',
             test_cfg=dict(
                 nms_pre=1000,
                 min_bbox_size=0,
                 score_thr=0.05,
                 nms=dict(type='nms', iou_threshold=0.5),
                 max_per_img=100)
             )

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='FixScaleResize',
         scale=800,
         keep_ratio=True,
         short_side_mode=True,
         backend='pillow',
         interpolation='bicubic'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'caption'))
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
