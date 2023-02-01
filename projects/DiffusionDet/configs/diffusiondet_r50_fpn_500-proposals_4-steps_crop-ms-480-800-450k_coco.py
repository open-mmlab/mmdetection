_base_ = 'diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py'  # noqa

# model settings
model = dict(bbox_head=dict(sampling_timesteps=4))
