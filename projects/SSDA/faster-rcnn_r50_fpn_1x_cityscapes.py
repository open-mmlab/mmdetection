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
                            data_prefix=dict(img='leftImg8bit_trainvaltest/leftImg8bit/train/'))
                        ))

val_dataloader = dict(batch_size=1,
                      dataset=dict(
                          data_prefix=dict(img='leftImg8bit_trainvaltest/leftImg8bit/val/'))
                      )
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

