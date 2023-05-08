_base_ = [
    './bytetrack_yolox_x_8xb4-80e_crowdhuman-mot20train_test-mot20test.py'
]

# fp16 settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
val_cfg = dict(type='ValLoop', fp16=True)
test_cfg = dict(type='TestLoop', fp16=True)
