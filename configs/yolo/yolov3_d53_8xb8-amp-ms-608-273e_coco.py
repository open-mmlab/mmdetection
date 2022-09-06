_base_ = './yolov3_d53_8xb8-ms-608-273e_coco.py'
# fp16 settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
