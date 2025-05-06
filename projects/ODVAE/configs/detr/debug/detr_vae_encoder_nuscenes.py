# detr_ldmencoder_nuscenes.py - Specific config for DETR with LDMEncoder backbone on NuScenes

_base_ = ['../detr_vae_encoder_nuscenes.py']

train_dataloader = _base_.train_dataloader


train_dataloader.persistent_workers = False
train_dataloader.num_workers = 0

val_dataloader = _base_.val_dataloader

val_dataloader.persistent_workers = False
val_dataloader.num_workers = 0

test_dataloader = _base_.test_dataloader

test_dataloader.persistent_workers = False
test_dataloader.num_workers = 0




