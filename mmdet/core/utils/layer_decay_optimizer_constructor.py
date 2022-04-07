# Copyright (c) OpenMMLab. All rights reserved.
import json

from mmcv import Registry
from mmcv.runner import OPTIMIZER_BUILDERS as MMCV_OPTIMIZER_BUILDERS
from mmcv.runner import DefaultOptimizerConstructor, get_dist_info

from ...utils import get_root_logger

OPTIMIZER_BUILDERS = Registry(
    'optimizer builder', parent=MMCV_OPTIMIZER_BUILDERS)


# Modified from
# https://github.com/facebookresearch/ConvNeXt/blob/main/object_detection/mmcv_custom/layer_decay_optimizer_constructor.py # noqa
def get_num_layer_layer_wise(var_name, num_max_layer=12):
    """Get the layer id to set the different learning rates in ``layer_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.

    Returns:
        int: The id number corresponding to different　learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    else:
        return num_max_layer + 1


def get_num_layer_stage_wise(var_name, num_max_layer):
    """Get the layer id to set the different learning rates in ``stage_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.
    Returns:
        int: The id number corresponding to different　learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        return 0
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        return stage_id + 1
    else:
        return num_max_layer - 1


@OPTIMIZER_BUILDERS.register_module()
class LearningRateDecayOptimizerConstructor(DefaultOptimizerConstructor):
    """Different learning rates are set for different layers of backbone."""

    def add_params(self, params, module, **kwargs):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """
        logger = get_root_logger()

        parameter_groups = {}
        logger.info(f'self.paramwise_cfg is {self.paramwise_cfg}')
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')
        logger.info('Build LearningRateDecayOptimizerConstructor  '
                    f'{decay_type} {decay_rate} - {num_layers}')
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            if decay_type == 'layer_wise':
                layer_id = get_num_layer_layer_wise(
                    name, self.paramwise_cfg.get('num_layers'))
                logger.info(f'set param {name} as id {layer_id}')
            elif decay_type == 'stage_wise':
                layer_id = get_num_layer_stage_wise(name, num_layers)
                logger.info(f'set param {name} as id {layer_id}')
            group_name = f'layer_{layer_id}_{group_name}'

            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            logger.info(f'Param groups = {json.dumps(to_display, indent=2)}')
        params.extend(parameter_groups.values())
