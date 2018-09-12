import torch.nn as nn

norm_cfg = {'BN': nn.BatchNorm2d, 'SyncBN': None, 'GN': None}


def build_norm_layer(cfg, num_features):
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    cfg_.setdefault('eps', 1e-5)
    layer_type = cfg_.pop('type')

    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    elif norm_cfg[layer_type] is None:
        raise NotImplementedError

    return norm_cfg[layer_type](num_features, **cfg_)
