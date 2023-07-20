# This is different from the TTA of official CenterNet.

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))

tta_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                # ``RandomFlip`` must be placed before ``RandomCenterCropPad``,
                # otherwise bounding box coordinates after flipping cannot be
                # recovered correctly.
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.)
            ],
            [
                dict(
                    type='RandomCenterCropPad',
                    ratios=None,
                    border=None,
                    mean=[0, 0, 0],
                    std=[1, 1, 1],
                    to_rgb=True,
                    test_mode=True,
                    test_pad_mode=['logical_or', 31],
                    test_pad_add_pix=1),
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'flip', 'flip_direction', 'border'))
            ]
        ])
]
