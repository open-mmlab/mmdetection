_base_ = './retinanet_r50_caffe_fpn_mstrain_3x_coco.py'

model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(
                    type='DropBlock',
                    drop_prob=0.05,
                    block_size=3,
                    postfix='_1'),
                stages=(False, False, True, True),
                position='after_conv1'),
            dict(
                cfg=dict(
                    type='DropBlock',
                    drop_prob=0.05,
                    block_size=3,
                    postfix='_2'),
                stages=(False, False, True, True),
                position='after_conv2'),
            dict(
                cfg=dict(
                    type='DropBlock',
                    drop_prob=0.05,
                    block_size=3,
                    postfix='_3'),
                stages=(False, False, True, True),
                position='after_conv3')
        ],
    )
)
