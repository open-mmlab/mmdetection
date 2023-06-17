_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/mot_challenge.py', '../_base_/default_runtime.py'
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    visualization=dict(type='TrackVisualizationHook', draw=False))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type='SyncBuffersHook')
]

detector = _base_.model
detector.pop('data_preprocessor')
detector.rpn_head.bbox_coder.update(dict(clip_border=False))
detector.roi_head.bbox_head.update(dict(num_classes=1))
detector.roi_head.bbox_head.bbox_coder.update(dict(clip_border=False))
detector['init_cfg'] = dict(
    type='Pretrained',
    checkpoint=  # noqa: E251
    'https://download.openmmlab.com/mmtracking/mot/'
    'faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth')  # noqa: E501
del _base_.model

model = dict(
    type='DeepSORT',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        rgb_to_bgr=False,
        pad_size_divisor=32),
    detector=detector,
    tracker=dict(
        type='SORTTracker',
        motion=dict(type='KalmanFilter', center_only=False),
        obj_score_thr=0.5,
        match_iou_thr=0.5,
        reid=None))

train_dataloader = None

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
