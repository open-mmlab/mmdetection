_base_ = [ './retinanet_r50_fpn_1x_voc0712.py' ]

total_epochs = 4
work_dir = './output'

find_unused_parameters = True

nncf_config = dict(
    input_info=dict(
        sample_size=[1, 3, 1000, 600]
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
