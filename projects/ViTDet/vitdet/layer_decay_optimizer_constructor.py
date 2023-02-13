import json
from typing import List

from torch import nn
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from mmengine.optim import DefaultOptimWrapperConstructor

from mmdet.registry import OPTIM_WRAPPER_CONSTRUCTORS


def get_num_layer_for_vit(var_name, num_max_layer, layer_sep=None):
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.patch_embed"):
        return 0
    elif var_name.startswith("backbone.blocks"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    elif var_name.startswith("backbone.layers"):
        assert layer_sep is not None
        split = var_name.split('.')
        start_id = layer_sep[int(split[2])]
        if split[3] == 'RC':
            return start_id
        return start_id + int(split[4]) + 1
    else:
        return num_max_layer - 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    
    def add_params(self, params: List[dict], module: nn.Module,
                   **kwargs) -> None:
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # get param-wise options
        logger = MMLogger.get_current_instance()
        parameter_groups = {}
        logger.info(f'self.paramwise_cfg is {self.paramwise_cfg}')
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_sep = self.paramwise_cfg.get('layer_sep', None)
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        logger.info('Build LayerDecayOptimizerConstructor  '
                    f'{layer_decay_rate} - {num_layers}')
        weight_decay = self.base_wd

        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(custom_keys.keys())

        for name, param in module.named_parameters():

            if not param.requires_grad:
                continue  # frozen weights

            if len(param.shape) == 1 or name.endswith(".bias") or ('pos_embed' in name) or ('cls_token' in name) or ('rel_pos_' in name):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            layer_id = get_num_layer_for_vit(name, num_layers, layer_sep)
            group_name = "layer_%d_%s" % (layer_id, group_name)

            # if the parameter match one of the custom keys, ignore other rules
            this_lr_multi = 1.
            for key in sorted_keys:
                if key in f'{name}':
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    this_lr_multi = lr_mult
                    group_name = "%s_%s" % (group_name, key)
                    break

            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [], 
                    "lr_scale": scale, 
                    "group_name": group_name, 
                    "lr": scale * self.base_lr * this_lr_multi, 
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"], 
                    "lr_scale": parameter_groups[key]["lr_scale"], 
                    "lr": parameter_groups[key]["lr"], 
                    "weight_decay": parameter_groups[key]["weight_decay"], 
                }
            logger.info(f'Param groups = {json.dumps(to_display, indent=2)}')

        params.extend(parameter_groups.values())
