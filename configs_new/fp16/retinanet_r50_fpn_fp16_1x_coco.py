_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'
# fp16 settings
fp16 = dict(loss_scale=512.)
work_dir = './work_dirs/retinanet_r50_fpn_fp16_1x'
