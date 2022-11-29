# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes
from ..assigners.assign_result import AssignResult
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


# TODO: replace these sampler after refactor
@TASK_UTILS.register_module()
class MaskBoxPseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result: AssignResult, pred_instances: InstanceData,
               gt_instances: InstanceData, *args, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Mask assigning results.
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``scores`` and ``masks`` predicted
                by the model.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``labels`` and ``masks``
                attributes.

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors

        pred_masks = pred_instances.bboxes
        gt_masks = gt_instances.masks
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = pred_masks.new_zeros(pred_masks.shape[0], dtype=torch.uint8)
        sampling_result = MaskBoxSamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            masks=pred_masks,
            gt_bboxes=gt_bboxes,
            gt_masks=gt_masks,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False)
        return sampling_result


class MaskBoxSamplingResult(SamplingResult):
    """Mask sampling result."""

    def __init__(self,
                 pos_inds: Tensor,
                 neg_inds: Tensor,
                 priors: Tensor,
                 masks: Tensor,
                 gt_bboxes: Tensor,
                 gt_masks: Tensor,
                 assign_result: AssignResult,
                 gt_flags: Tensor,
                 avg_factor_with_neg: bool = True) -> None:
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.num_pos = max(pos_inds.numel(), 1)
        self.num_neg = max(neg_inds.numel(), 1)
        self.avg_factor = self.num_pos + self.num_neg \
            if avg_factor_with_neg else self.num_pos

        self.pos_masks = masks[pos_inds]
        self.neg_masks = masks[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_masks.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        self.pos_priors = priors[pos_inds]
        self.neg_priors = priors[neg_inds]

        self.pos_gt_labels = assign_result.labels[pos_inds]
        box_dim = gt_bboxes.box_dim if isinstance(gt_bboxes, BaseBoxes) else 4
        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = gt_bboxes.view(-1, box_dim)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, box_dim)
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds.long()]

        if gt_masks.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_masks = torch.empty_like(gt_masks)
        else:
            self.pos_gt_masks = gt_masks[self.pos_assigned_gt_inds, :]

    @property
    def masks(self) -> Tensor:
        """torch.Tensor: concatenated positive and negative masks."""
        return torch.cat([self.pos_masks, self.neg_masks])

    def __nice__(self) -> str:
        data = self.info.copy()
        data['pos_masks'] = data.pop('pos_masks').shape
        data['neg_masks'] = data.pop('neg_masks').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'

    @property
    def info(self) -> dict:
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_masks': self.pos_masks,
            'neg_masks': self.neg_masks,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }
