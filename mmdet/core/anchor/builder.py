from mmcv.utils import Registry, build_from_cfg

ANCHOR_GENERATORS = Registry('Anchor generator')


def build_anchor_generator(cfg, default_args=None):
    return build_from_cfg(cfg, ANCHOR_GENERATORS, default_args)
