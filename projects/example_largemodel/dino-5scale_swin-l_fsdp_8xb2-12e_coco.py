from mmengine.config import read_base
from mmdet.models.backbones.swin import SwinBlock
from mmdet.models.layers.transformer.deformable_detr_layers import DeformableDetrTransformerEncoderLayer

with read_base():
    from mmdet.configs.dino.dino_5scale_swin_l_8xb2_12e_coco import *

custom_imports = dict(
    imports=['projects.example_largemodel'], allow_failed_imports=False)

from projects.example_largemodel import layer_auto_wrap_policy, checkpoint_check_fn

layer_cls = (SwinBlock, DeformableDetrTransformerEncoderLayer)

model.update(dict(backbone=dict(with_cp=False)))

runner_type = 'FlexibleRunner'
strategy = dict(
    type='FSDPStrategy',
    gradient_checkpoint=dict(check_fn=dict(type=checkpoint_check_fn, layer_cls=layer_cls)),
    model_wrapper=dict(
        auto_wrap_policy=dict(
            type=layer_auto_wrap_policy,
            layer_cls=layer_cls)))
