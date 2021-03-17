from mmcv.utils import Registry, build_from_cfg

TRANSFORMER = Registry('Transformer')
POSITIONAL_ENCODING = Registry('Position encoding')
ATTENTION = Registry('Attention')
TRANSFORMERLAYER = Registry('TransformerLayer')
TRANSFORMERCODER = Registry('TransformerCoder')


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_transformerlayer(cfg, default_args=None):
    """Builder for transformer layer."""
    return build_from_cfg(cfg, TRANSFORMERLAYER, default_args)


def build_transformercoder(cfg, default_args=None):
    """Builder for transformer coder."""
    return build_from_cfg(cfg, TRANSFORMERCODER, default_args)
