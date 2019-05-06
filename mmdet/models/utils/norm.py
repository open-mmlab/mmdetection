import torch.nn as nn


norm_cfg = {
    # format: layer_type: (abbreviation, module)
    'BN': ('bn', nn.BatchNorm2d),
    'SyncBN': ('bn', nn.SyncBatchNorm),
    'GN': ('gn', nn.GroupNorm),
    # and potentially 'SN'
}


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            frozen (bool): [optional] whether stop gradient updates
                of norm layer, it is helpful to set frozen mode
                in backbone's norms.
        num_features (int): number of channels from input
        postfix (int, str): appended into norm abbreation to
            create named layer.

    Returns:
        name (str): abbreation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    frozen = cfg_.pop('frozen', False)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer.ddp_gpu_size = 1
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    if frozen:
        for param in layer.parameters():
            param.requires_grad = False

    return name, layer


def convert_batchnorm(module):
    module_output = module
    if isinstance(module, nn.SyncBatchNorm):
        module_output = nn.BatchNorm2d(module.num_features,
                                       module.eps, module.momentum,
                                       module.affine,
                                       module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, convert_batchnorm(child))
    del module
    return module_output
