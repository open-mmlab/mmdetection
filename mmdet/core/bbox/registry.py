from mmcv.utils import Registry, build_from_cfg

BBOX_ASSIGNERS = Registry('bbox_assigner')
BBOX_SAMPLERS = Registry('bbox_sampler')
BBOX_CODERS = Registry('bbox_coder')


def build_assigner(cfg, **default_args):
    from .assigners import BaseAssigner
    if isinstance(cfg, BaseAssigner):
        return cfg
    return build_from_cfg(cfg, BBOX_ASSIGNERS, default_args)


def build_sampler(cfg, **default_args):
    from .samplers import BaseSampler
    if isinstance(cfg, BaseSampler):
        return cfg
    return build_from_cfg(cfg, BBOX_SAMPLERS, default_args)


def build_bbox_coder(cfg, **default_args):
    from .coder import BaseBBoxCoder
    if isinstance(cfg, BaseBBoxCoder):
        return cfg
    return build_from_cfg(cfg, BBOX_CODERS, default_args)
