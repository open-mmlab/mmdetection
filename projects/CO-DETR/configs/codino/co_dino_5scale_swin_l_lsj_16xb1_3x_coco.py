_base_ = ['co_dino_5scale_swin_l_lsj_16xb1_1x_coco.py']

model = dict(backbone=dict(drop_path_rate=0.5))

param_scheduler = [dict(milestones=[30])]
train_cfg = dict(max_epochs=36)
