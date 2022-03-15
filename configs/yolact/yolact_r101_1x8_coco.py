_base_ = './yolact_r50_1x8_coco.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

# NOTE: This is for automatically scaling LR, USER CAN'T CHANGE THIS VALUE
default_batch_size = 8  # (1 GPUs) x (8 samples per GPU)
