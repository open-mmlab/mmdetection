_base_ = [ '../../pascal_voc/ssd300_voc0712.py' ]

optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=5e-4)

total_epochs = 2

work_dir = './output'
load_from = '../original_mmdetection/mmdetection/ssd300_voc_vgg16_caffe_240e_20190501-7160d09a.pth'

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
