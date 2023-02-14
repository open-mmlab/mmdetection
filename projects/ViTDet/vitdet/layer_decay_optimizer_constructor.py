# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import List

import torch.nn as nn
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from mmengine.optim import DefaultOptimWrapperConstructor

from mmdet.registry import OPTIM_WRAPPER_CONSTRUCTORS


def get_layer_id_for_vit(var_name, max_layer_id):
    """Get the layer id to set the different learning rates in ``layer_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_layer_id (int): Maximum layer id.
    Returns:
        int: The id number corresponding to different learning rate in
        ``LayerDecayOptimizerConstructor``.
    """
    if var_name.startswith('backbone'):
        if 'patch_embed' in var_name or 'pos_embed' in var_name:
            return 0
        elif '.blocks.' in var_name:
            layer_id = int(var_name.split('.')[2]) + 1
            return layer_id
        else:
            return max_layer_id + 1
    else:
        return max_layer_id + 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    # Different learning rates are set for different layers of backbone.
    # Note: Currently, this optimizer constructor is built for ViT.

    def add_params(self, params: List[dict], module: nn.Module,
                   **kwargs) -> None:
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """
        logger = MMLogger.get_current_instance()

        parameter_groups = {}
        logger.info(f'self.paramwise_cfg is {self.paramwise_cfg}')
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')
        logger.info('Build LayerDecayOptimizerConstructor  '
                    f'{decay_type} {decay_rate} - {num_layers}')
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if name.startswith('backbone.blocks') and 'norm' in name:
                group_name = 'no_decay'
                this_weight_decay = 0.
            elif 'pos_embed' in name:
                group_name = 'no_decay_pos_embed'
                this_weight_decay = 0
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            layer_id = get_layer_id_for_vit(
                name, self.paramwise_cfg.get('num_layers'))
            logger.info(f'set param {name} as id {layer_id}')

            group_name = f'layer_{layer_id}_{group_name}'
            this_lr_multi = 1.

            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - 1 - layer_id)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr * this_lr_multi,
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
