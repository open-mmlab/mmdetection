_base_ = './glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py'

model = dict(bbox_head=dict(early_fuse=True, use_checkpoint=True))

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_b_mmdet-6dfbd102.pth'  # noqa

optim_wrapper = dict(
    optimizer=dict(lr=0.00001),
    clip_grad=dict(_delete_=True, max_norm=1, norm_type=2))
