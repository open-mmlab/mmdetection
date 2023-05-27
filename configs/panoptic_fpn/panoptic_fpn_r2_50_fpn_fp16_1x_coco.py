_base_ = './panoptic_fpn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        type='Res2Net',
        depth=50,
        scales=4,
        base_width=26,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='res2net50_v1b_26w_4s-3cf99910.pth')))

fp16 = dict(loss_scale='dynamic')
