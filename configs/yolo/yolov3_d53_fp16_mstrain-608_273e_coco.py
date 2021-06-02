_base_ = './yolov3_d53_mstrain-608_273e_coco.py'
# fp16 settings
fp16 = dict(loss_scale='dynamic')
