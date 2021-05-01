import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg

TRANSFORMER = Registry('Transformer')
LINEAR_LAYERS = Registry('linear layers')


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


LINEAR_LAYERS.register_module('Linear', module=nn.Linear)


def build_linear_layer(cfg, default_args=None):
    """Builder for Linear Layer."""
    return build_from_cfg(cfg, LINEAR_LAYERS, default_args)
