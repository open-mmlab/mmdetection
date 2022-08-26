# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional

import numpy as np
from mmcv.transforms import BaseTransform, Compose
from mmcv.transforms.utils import cache_randomness

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MultiBranch(BaseTransform):
    r"""Multiple branch pipeline wrapper.

    Generate multiple data-augmented versions of the same image.
    `MultiBranch` needs to specify the branch names of all
    pipelines of the dataset, perform corresponding data augmentation
    for the current branch, and return None for other branches,
    which ensures the uniformity of return value.

    Args:
        branch_field (list): List of branch names.
        branch_pipelines (dict): Dict of different pipeline configs
            to be composed.

    Examples:
        >>> branch_field = ['sup', 'unsup_teacher', 'unsup_student']
        >>> sup_pipeline = [
        >>>     dict(type='LoadImageFromFile',
        >>>         file_client_args=file_client_args),
        >>>     dict(type='LoadAnnotations', with_bbox=True),
        >>>     dict(type='RandomResize', scale=scale, keep_ratio=True),
        >>>     dict(type='RandomFlip', prob=0.5),
        >>>     dict(
        >>>         type='MultiBranch',
        >>>         branch_field=branch_field,
        >>>         sup=dict(type='PackDetInputs'))
        >>>     ]
        >>> weak_pipeline = [
        >>>     dict(type='LoadImageFromFile',
        >>>         file_client_args=file_client_args),
        >>>     dict(type='LoadAnnotations', with_bbox=True),
        >>>     dict(type='RandomResize', scale=scale, keep_ratio=True),
        >>>     dict(type='RandomFlip', prob=0.0),
        >>>     dict(
        >>>         type='MultiBranch',
        >>>         branch_field=branch_field,
        >>>         sup=dict(type='PackDetInputs'))
        >>>     ]
        >>> strong_pipeline = [
        >>>     dict(type='LoadImageFromFile',
        >>>         file_client_args=file_client_args),
        >>>     dict(type='LoadAnnotations', with_bbox=True),
        >>>     dict(type='RandomResize', scale=scale, keep_ratio=True),
        >>>     dict(type='RandomFlip', prob=1.0),
        >>>     dict(
        >>>         type='MultiBranch',
        >>>         branch_field=branch_field,
        >>>         sup=dict(type='PackDetInputs'))
        >>>     ]
        >>> unsup_pipeline = [
        >>>     dict(type='LoadImageFromFile',
        >>>         file_client_args=file_client_args),
        >>>     dict(type='LoadEmptyAnnotations'),
        >>>     dict(
        >>>         type='MultiBranch',
        >>>         branch_field=branch_field,
        >>>         unsup_teacher=weak_pipeline,
        >>>         unsup_student=strong_pipeline)
        >>>     ]

    """

    def __init__(self, branch_field: List[str],
                 **branch_pipelines: dict) -> None:
        self.branch_field = branch_field
        self.branch_pipelines = {
            branch: Compose(pipeline)
            for branch, pipeline in branch_pipelines.items()
        }

    def transform(self, results: dict) -> dict:
        """Transform function to apply transforms sequentially.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict:

            - 'inputs' (Dict[str, obj:`torch.Tensor`]): The forward data of
                models from different branches.
            - 'data_sample' (Dict[str,obj:`DetDataSample`]): The annotation
                info of the sample from different branches.
        """
        multi_results = {}
        for branch in self.branch_field:
            multi_results[branch] = {'inputs': None, 'data_samples': None}
        for branch, pipeline in self.branch_pipelines.items():
            branch_results = pipeline(copy.deepcopy(results))
            # If one branch pipeline returns None,
            # it will sample another data from dataset.
            if branch_results is None:
                return None
            multi_results[branch] = branch_results

        format_results = {}
        for branch, results in multi_results.items():
            for key in results.keys():
                if format_results.get(key, None) is None:
                    format_results[key] = {branch: results[key]}
                else:
                    format_results[key][branch] = results[key]
        return format_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(branch_pipelines={list(self.branch_pipelines.keys())})'
        return repr_str


@TRANSFORMS.register_module()
class RandomOrder(Compose):
    """Shuffle the transform Sequence."""

    @cache_randomness
    def _random_permutation(self):
        return np.random.permutation(len(self.transforms))

    def transform(self, results: Dict) -> Optional[Dict]:
        """Transform function to apply transforms in random order.

        Args:
            results (dict): A result dict contains the results to transform.

        Returns:
            dict or None: Transformed results.
        """
        inds = self._random_permutation()
        for idx in inds:
            t = self.transforms[idx]
            results = t(results)
            if results is None:
                return None
        return results

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'{t.__class__.__name__}, '
        format_string += ')'
        return format_string
