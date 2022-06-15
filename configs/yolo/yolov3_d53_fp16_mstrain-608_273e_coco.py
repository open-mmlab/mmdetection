_base_ = './yolov3_d53_mstrain-608_273e_coco.py'
# fp16 settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
