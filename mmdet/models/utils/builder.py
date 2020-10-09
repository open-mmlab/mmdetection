from mmcv.utils import Registry, build_from_cfg

TRANSFORMER = Registry('Transformer')
POSITION_ENCODING = Registry('Position encoding')


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


def build_position_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, POSITION_ENCODING, default_args)
