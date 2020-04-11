from mmdet.utils import build_from_cfg
from .assigners import BaseAssigner
from .registry import BBOX_ASSIGNERS, BBOX_SAMPLERS
from .samplers import BaseSampler


def build_assigner(cfg, **default_args):
    if isinstance(cfg, BaseAssigner):
        return cfg
    return build_from_cfg(cfg, BBOX_ASSIGNERS, default_args)


def build_sampler(cfg, **default_args):
    if isinstance(cfg, BaseSampler):
        return cfg
    return build_from_cfg(cfg, BBOX_SAMPLERS, default_args)


# TODO remove this function in anchor_target in the future
def assign_and_sample(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels, cfg):
    bbox_assigner = build_assigner(cfg.assigner)
    bbox_sampler = build_sampler(cfg.sampler)
    assign_result = bbox_assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore,
                                         gt_labels)
    sampling_result = bbox_sampler.sample(assign_result, bboxes, gt_bboxes,
                                          gt_labels)
    return assign_result, sampling_result
