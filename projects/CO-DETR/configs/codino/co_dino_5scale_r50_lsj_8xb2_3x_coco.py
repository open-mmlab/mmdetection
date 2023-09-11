_base_ = ['co_dino_5scale_r50_lsj_8xb2_1x_coco.py']

param_scheduler = [dict(milestones=[30])]
train_cfg = dict(max_epochs=36)
