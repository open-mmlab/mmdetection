import warnings

from mmcv.utils import Registry, build_from_cfg

PROIRS_GENERATORS = Registry('Priors generator')

ANCHOR_GENERATORS = PROIRS_GENERATORS


def build_priors_generator(cfg, default_args=None):
    return build_from_cfg(cfg, ANCHOR_GENERATORS, default_args)


def build_anchor_generator(cfg, default_args=None):
    warnings.warn(
        '``build_anchor_generator`` would be deprecated soon, please use '
        '``build_priors_generator`` ')
    return build_priors_generator(cfg, default_args=default_args)
