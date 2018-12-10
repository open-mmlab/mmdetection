import torch.nn as nn

norm_cfg = {'BN': nn.BatchNorm2d, 'SyncBN': None, 'GN': nn.GroupNorm}


def build_norm_layer(cfg, num_features):
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    cfg_.setdefault('eps', 1e-5)
    layer_type = cfg_.pop('type')

    # args name matching
    if layer_type == 'GN':
        cfg_.setdefault('num_channels', num_features)
    else:
        cfg_.setdefault('num_features', num_features)

    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    elif norm_cfg[layer_type] is None:
        raise NotImplementedError

    return norm_cfg[layer_type](**cfg_)
