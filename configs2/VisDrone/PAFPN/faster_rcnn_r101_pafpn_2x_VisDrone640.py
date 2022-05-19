_base_='../base/faster_rcnn_r50_fpn_2x_VisDrone640.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5)
)
evaluation = dict(interval=2, metric='bbox',save_best='bbox_mAP_s')