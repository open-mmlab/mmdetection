_base_ = [ './retinanet_r50_fpn_1x_voc0712.py' ]

work_dir = './output'

find_unused_parameters = True

nncf_config = dict(
    input_info=dict(
        sample_size=[1, 3, 1000, 600]
    ),
    compression=[
        dict(
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
        dict(
            algorithm='magnitude_sparsity',
            params=dict(
                schedule='multistep',
                multistep_sparsity_levels=[
                    0.3,
                    0.5,
                    0.7
                ],
                multistep_steps=[
                    40,
                    80
                ]
            )
        )
    ],
    log_dir=work_dir
)
