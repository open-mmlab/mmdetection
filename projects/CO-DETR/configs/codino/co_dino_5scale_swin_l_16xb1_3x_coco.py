_base_ = ['co_dino_5scale_swin_l_16xb1_1x_coco.py']
# model settings
model = dict(backbone=dict(drop_path_rate=0.6))

param_scheduler = [dict(milestones=[30])]
train_cfg = dict(max_epochs=36)
