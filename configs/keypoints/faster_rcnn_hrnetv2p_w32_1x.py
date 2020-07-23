_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/wflw.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        _delete_=True,
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))),
    neck=dict(
        _delete_=True,
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256),
    rpn_head=None,
    roi_head=dict(
        _delete_=True,
        type='KeypointRoIHead',
        output_heatmaps=False,
        keypoint_head=dict(
            type='HRNetKeypointHead',
            num_convs=8,
            in_channels=256,
            features_size=[256, 256, 256, 256],
            conv_out_channels=512,
            num_keypoints=98,
            loss_keypoint=dict(type='MSELoss', loss_weight=50.0)),
        keypoint_decoder=dict(type='HeatmapDecodeOneKeypoint', upscale=4)))
test_cfg = dict(rcnn=dict(score_thr=-1))

optimizer = dict(lr=0.002)
lr_config = dict(step=[40, 55])
total_epochs = 60
