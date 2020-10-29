_base_ = ['../../ssd/ssd300_coco.py']

optimizer = dict(type='SGD', lr=2e-5, momentum=0.9, weight_decay=5e-4)

work_dir = './output'
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth'

find_unused_parameters = True

nncf_config = dict(
    input_info=dict(
        sample_size=[1, 3, 300, 300]
    ),
    compression=dict(
        algorithm='quantization',
        initializer=dict(
            range=dict(
                num_init_steps=10
            ),
            batchnorm_adaptation=dict(
                num_bn_adaptation_steps=30,
            )
        )
    ),
    log_dir=work_dir
)
