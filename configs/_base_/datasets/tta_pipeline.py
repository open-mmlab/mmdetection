file_client_args = dict(backend='disk')
tta_pipeline = [
        dict(type='LoadImageFromFile', file_client_args=file_client_args),
        dict(type='TestTimeAug',
             transforms=[
                 [dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                  dict(type='Resize', scale=(1333, 800), keep_ratio=True)],
                 [dict(type='RandomFlip', prob=1.),
                  dict(type='RandomFlip', prob=0.)],
                 [dict(type='LoadAnnotations', with_bbox=True)],
                 [dict(type='PackDetInputs',
                       meta_keys=('img_id', 'img_path', 'ori_shape',
                                  'img_shape', 'scale_factor', 'flip',
                                  'flip_direction'))]])
    ]
