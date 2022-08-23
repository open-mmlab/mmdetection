_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'
# model settings
model = dict(
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_input',
            num_outs=5),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5,
            refine_level=1,
            refine_type='non_local')
    ],
    bbox_head=dict(
        loss_bbox=dict(
            _delete_=True,
            type='BalancedL1Loss',
            alpha=0.5,
            gamma=1.5,
            beta=0.11,
            loss_weight=1.0)))
