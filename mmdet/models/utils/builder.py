from mmcv.utils import Registry, build_from_cfg

TRANSFORMER = Registry('Transformer')


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)
