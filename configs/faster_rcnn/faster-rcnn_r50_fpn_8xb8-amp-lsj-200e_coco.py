_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../common/lsj-200e_coco-detection.py'
]
image_size = (1024, 1024)
batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]

model = dict(data_preprocessor=dict(batch_augments=batch_augments))

train_dataloader = dict(batch_size=8, num_workers=4)
# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.02 * 4, momentum=0.9, weight_decay=0.00004))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
