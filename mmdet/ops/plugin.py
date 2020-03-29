from .context_block import ContextBlock
from .generalized_attention import GeneralizedAttention
from .non_local import NonLocal2D

plugin_cfg = {
    # format: layer_type: (abbreviation, module)
    'ContextBlock': ('context_block', ContextBlock),
    'GeneralizedAttention': ('gen_attention_block', GeneralizedAttention),
    'NonLocal2D': ('nonlocal_block', NonLocal2D)
}


def build_plugin_layer(cfg, *args, **kwargs):
    """ Build plugin layer
    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify plugin layer type.
            layer args: args needed to instantiate a plugin layer.
    Returns:
        layer (nn.Module): created plugin layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in plugin_cfg:
        raise KeyError('Unrecognized plugin type {}'.format(layer_type))
    else:
        name, plugin_layer = plugin_cfg[layer_type]

    layer = plugin_layer(*args, **kwargs, **cfg_)

    return name, layer
