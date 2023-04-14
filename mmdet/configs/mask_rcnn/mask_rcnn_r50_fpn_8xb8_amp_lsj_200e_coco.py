if '_base_':
    from .._base_.models.mask_rcnn_r50_fpn import *
    from ..common.lsj_100e_coco_instance import *
from mmdet.models.data_preprocessors.data_preprocessor import BatchFixedSizePad
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper
from torch.optim.sgd import SGD

image_size = (1024, 1024)
batch_augments = [dict(type=BatchFixedSizePad, size=image_size, pad_mask=True)]

model.merge(dict(data_preprocessor=dict(batch_augments=batch_augments)))

train_dataloader.merge(dict(batch_size=8, num_workers=4))
# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper.merge(
    dict(
        type=AmpOptimWrapper,
        optimizer=dict(
            type=SGD, lr=0.02 * 4, momentum=0.9, weight_decay=0.00004)))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=64))
