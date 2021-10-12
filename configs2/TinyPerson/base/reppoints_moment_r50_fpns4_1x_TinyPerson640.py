_base_ = [
    "./reppoints_moment_r50_fpn_1x_TinyPerson640.py"
]

model = dict(
    neck=dict(
        type='FPN',
        start_level=0,  # 1
    ),
    bbox_head=dict(
        point_strides=[4, 8, 16, 32, 64],  # [8, 16, 32, 64, 128]
        point_base_scale=2,  # 4
    ),
    # training and testing settings
    train_cfg=dict(
        init=dict(
            assigner=dict(type='PointAssigner', scale=2, pos_num=1),  # scale=4
        )
    )
)

