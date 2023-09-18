# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.structures.bbox import BaseBoxes, cat_boxes
from mmdet.utils import util_mixins
from mmdet.utils.util_random import ensure_rng


def random_boxes(num=1, scale=1, rng=None):
    """Simple version of ``kwimage.Boxes.random``

    Returns:
        Tensor: shape (n, 4) in x1, y1, x2, y2 format.

    References:
        https://gitlab.kitware.com/computer-vision/kwimage/blob/master/kwimage/structs/boxes.py#L1390

    Example:
        >>> num = 3
        >>> scale = 512
        >>> rng = 0
        >>> boxes = random_boxes(num, scale, rng)
        >>> print(boxes)
        tensor([[280.9925, 278.9802, 308.6148, 366.1769],
                [216.9113, 330.6978, 224.0446, 456.5878],
                [405.3632, 196.3221, 493.3953, 270.7942]])
    """
    rng = ensure_rng(rng)

    tlbr = rng.rand(num, 4).astype(np.float32)

    tl_x = np.minimum(tlbr[:, 0], tlbr[:, 2])
    tl_y = np.minimum(tlbr[:, 1], tlbr[:, 3])
    br_x = np.maximum(tlbr[:, 0], tlbr[:, 2])
    br_y = np.maximum(tlbr[:, 1], tlbr[:, 3])

    tlbr[:, 0] = tl_x * scale
    tlbr[:, 1] = tl_y * scale
    tlbr[:, 2] = br_x * scale
    tlbr[:, 3] = br_y * scale

    boxes = torch.from_numpy(tlbr)
    return boxes


class SamplingResult(util_mixins.NiceRepr):
    """Bbox sampling result.

    Args:
        pos_inds (Tensor): Indices of positive samples.
        neg_inds (Tensor): Indices of negative samples.
        priors (Tensor): The priors can be anchors or points,
            or the bboxes predicted by the previous stage.
        gt_bboxes (Tensor): Ground truth of bboxes.
        assign_result (:obj:`AssignResult`): Assigning results.
        gt_flags (Tensor): The Ground truth flags.
        avg_factor_with_neg (bool):  If True, ``avg_factor`` equal to
            the number of total priors; Otherwise, it is the number of
            positive priors. Defaults to True.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.models.task_modules.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_inds': tensor([1,  2,  3,  5,  6,  7,  8,
                                9, 10, 11, 12, 13]),
            'neg_priors': torch.Size([12, 4]),
            'num_gts': 1,
            'num_neg': 12,
            'num_pos': 1,
            'avg_factor': 13,
            'pos_assigned_gt_inds': tensor([0]),
            'pos_inds': tensor([0]),
            'pos_is_gt': tensor([1], dtype=torch.uint8),
            'pos_priors': torch.Size([1, 4])
        })>
    """

    def __init__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 pos_inds: Tensor,
                 neg_inds: Tensor,
                 avg_factor_with_neg: bool = True) -> None:
        self.pred_instances = pred_instances
        self.gt_instances = gt_instances
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.num_pos = max(pos_inds.numel(), 1)
        self.num_neg = max(neg_inds.numel(), 1)
        self.avg_factor_with_neg = avg_factor_with_neg
        self.avg_factor = self.num_pos + self.num_neg \
            if avg_factor_with_neg else self.num_pos

        self.pos_assigned_gt_inds = pred_instances.gt_inds[pos_inds] - 1

    def update_sampling_result(self,
                               pos_inds: Tensor,
                               neg_inds: Tensor,
                               avg_factor_with_neg: bool = True) -> None:
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.num_pos = max(pos_inds.numel(), 1)
        self.num_neg = max(neg_inds.numel(), 1)
        self.avg_factor_with_neg = avg_factor_with_neg
        self.avg_factor = self.num_pos + self.num_neg \
            if avg_factor_with_neg else self.num_pos

        self.pos_assigned_gt_inds = self.pred_instances.gt_inds[pos_inds] - 1

    @property
    def num_gts(self):
        return len(self.gt_instances)

    @property
    def pos_priors(self):
        return self.pred_instances.priors[self.pos_inds]

    @property
    def neg_priors(self):
        return self.pred_instances.priors[self.neg_inds]

    @property
    def priors(self):
        """torch.Tensor: concatenated positive and negative priors"""
        return cat_boxes([self.pos_priors, self.neg_priors])

    @property
    def max_overlaps(self):
        return self.pred_instances.max_overlaps

    @property
    def gt_inds(self):
        return self.pred_instances.gt_inds

    @property
    def pos_pred_instances(self):
        return self.pred_instances[self.pos_inds]

    @property
    def pos_gt_instances(self):
        return self.gt_instances[self.pos_assigned_gt_inds]

    @property
    def pos_gt_bboxes(self):
        gt_bboxes = self.gt_instances.bboxes
        box_dim = gt_bboxes.box_dim if isinstance(gt_bboxes, BaseBoxes) else 4
        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = gt_bboxes.view(-1, box_dim)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, box_dim)
            pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds.long()]
        return pos_gt_bboxes

    @property
    def pos_gt_labels(self):
        return self.gt_instances.labels[self.pos_assigned_gt_inds]

    @property
    def pos_is_gt(self):
        return self.pred_instances.gt_flags[self.pos_inds]

    @property
    def pos_masks(self):
        return self.pred_instances.masks[self.pos_inds]

    @property
    def neg_masks(self):
        return self.pred_instances.masks[self.neg_inds]

    @property
    def pos_gt_masks(self):
        gt_masks = self.gt_instances.masks
        if gt_masks.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            pos_gt_masks = torch.empty_like(gt_masks)
        else:
            pos_gt_masks = gt_masks[self.pos_assigned_gt_inds, :]
        return pos_gt_masks

    def get_pos(self, name):
        assert name in self.pred_instances
        return self.pred_instances.get(name)[self.pos_inds]

    def get_neg(self, name):
        assert name in self.pred_instances
        return self.pred_instances.get(name)[self.neg_inds]

    def to(self, device):
        """Change the device of the data inplace.

        Example:
            >>> self = SamplingResult.random()
            >>> print(f'self = {self.to(None)}')
            >>> # xdoctest: +REQUIRES(--gpu)
            >>> print(f'self = {self.to(0)}')
        """
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, (torch.Tensor, BaseBoxes)):
                _dict[key] = value.to(device)
        return self

    def add_gt_(self, gt_instances):
        """Add ground truth as assigned results.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
        """
        gt_bboxes = gt_instances.bboxes
        priors = self.pred_instances.priors
        if (isinstance(gt_bboxes, BaseBoxes)
                and isinstance(priors, BaseBoxes)):
            gt_bboxes = gt_bboxes.convert_to(type(priors))
        else:
            gt_bboxes = gt_bboxes
        num_gts = len(gt_instances)

        add_gt_instances = InstanceData()
        add_gt_instances.priors = gt_bboxes
        add_gt_instances.gt_inds = torch.arange(
            1,
            num_gts + 1,
            dtype=torch.long,
            device=gt_instances.labels.device)
        add_gt_instances.gt_flags = priors.new_ones((num_gts, ),
                                                    dtype=torch.uint8)
        add_gt_instances.max_overlaps = self.max_overlaps.new_ones(num_gts)
        # TODO: find a more elegant way to match the elements in
        #  `self.pred_instances`
        for k, v in self.pred_instances.items():
            if k not in add_gt_instances:
                shape = list(v.shape)
                shape[0] = num_gts
                add_gt_instances[k] = priors.new_ones(shape)
        concat_instances = self.pred_instances.cat(
            [add_gt_instances, self.pred_instances])
        self.pred_instances = concat_instances

    def __nice__(self):
        data = self.info.copy()
        data['pos_priors'] = data.pop('pos_priors').shape
        data['neg_priors'] = data.pop('neg_priors').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_priors': self.pos_priors,
            'neg_priors': self.neg_priors,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
            'num_pos': self.num_pos,
            'num_neg': self.num_neg,
            'avg_factor': self.avg_factor
        }

    @classmethod
    def random(cls, rng=None, **kwargs):
        """
        Args:
            rng (None | int | numpy.random.RandomState): seed or state.
            kwargs (keyword arguments):
                - num_preds: Number of predicted boxes.
                - num_gts: Number of true boxes.
                - p_ignore (float): Probability of a predicted box assigned to
                    an ignored truth.
                - p_assigned (float): probability of a predicted box not being
                    assigned.

        Returns:
            :obj:`SamplingResult`: Randomly generated sampling result.

        Example:
            >>> from mmdet.models.task_modules.samplers.sampling_result import *  # NOQA
            >>> self = SamplingResult.random()
            >>> print(self.__dict__)
        """
        from mmengine.structures import InstanceData

        from mmdet.models.task_modules.assigners import AssignResult
        from mmdet.models.task_modules.samplers import RandomSampler
        rng = ensure_rng(rng)

        # make probabilistic?
        num = 32
        pos_fraction = 0.5
        neg_pos_ub = -1

        assign_result = AssignResult.random(rng=rng, **kwargs)

        # Note we could just compute an assignment
        priors = random_boxes(assign_result.num_preds, rng=rng)
        gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
        gt_labels = torch.randint(
            0, 5, (assign_result.num_gts, ), dtype=torch.long)

        pred_instances = InstanceData()
        pred_instances.priors = priors

        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels

        add_gt_as_proposals = True

        sampler = RandomSampler(
            num,
            pos_fraction,
            neg_pos_ub=neg_pos_ub,
            add_gt_as_proposals=add_gt_as_proposals,
            rng=rng)
        self = sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        return self
