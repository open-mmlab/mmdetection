from mmcv.utils import Registry, build_from_cfg

TRANSFORMER = Registry('Transformer')
positional_encoding = Registry('Position encoding')


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, positional_encoding, default_args)
