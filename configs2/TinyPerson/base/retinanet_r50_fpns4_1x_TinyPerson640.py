_base_ = [
    './retinanet_r50_fpn_1x_TinyPerson640.py'
]

# model settings
model = dict(
    neck=dict(
        start_level=0,  # start_level=1,
        # add_extra_convs='on_input',  # note
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,  # 80
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=2,    # 4
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]  # [8, 16, 32, 64, 128]
        )
    )
)
