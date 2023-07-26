_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/cityscapes_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py', 'mmdet::_base_/default_runtime.py'
]
batch_size = 8
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=8
        )
    )
)

train_dataloader = dict(batch_size=batch_size,
                        dataset=dict(dataset=dict(
                            ann_file='annotations/instancesonly_filtered_gtFine_foggy_beta0.02_train.json',
                            data_prefix=dict(img='leftImg8bit_foggy/train/'))
                        ))

val_dataloader = dict(batch_size=1,
                      dataset=dict(
                          ann_file='annotations/instancesonly_filtered_gtFine_foggy_beta0.02_val.json',
                          data_prefix=dict(img='leftImg8bit_foggy/val/'))
                      )

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/cityscapes/annotations/instancesonly_filtered_gtFine_foggy_beta0.02_val.json',
    metric='bbox')

test_evaluator = val_evaluator


# gloo - multi-gpu
env_cfg = dict(dist_cfg=dict(backend='gloo'))

vis_backends = [
    dict(type='LocalVisBackend', _scope_='mmdet'),
    dict(type='WandbVisBackend', _scope_='mmdet')
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='WandbVisBackend')],
    name='visualizer',
    _scope_='mmdet')

