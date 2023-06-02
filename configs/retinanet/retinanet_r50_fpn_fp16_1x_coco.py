_base_ = './retinanet_r50_fpn_1x_coco.py'
# fp16 settings
fp16 = dict(loss_scale=512.)

# set grad_norm for stability during mixed-precision training
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
