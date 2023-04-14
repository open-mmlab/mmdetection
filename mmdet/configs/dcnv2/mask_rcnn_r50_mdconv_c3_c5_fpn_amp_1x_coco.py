if '_base_':
    from ..mask_rcnn.mask_rcnn_r50_fpn_1x_coco import *
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper

model.merge(
    dict(
        backbone=dict(
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, True, True, True))))

# MMEngine support the following two ways, users can choose
# according to convenience
# optim_wrapper = dict(type=AmpOptimWrapper)
optim_wrapper.type = 'AmpOptimWrapper'
