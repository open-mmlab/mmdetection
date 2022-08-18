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

    Args:
        branch_pipelines (dict): Dict of different pipeline configs
            to be composed.

    """

    def __init__(self, **branch_pipelines: dict) -> None:
        self.branch_pipelines = {
            branch: Compose(pipeline)
            for branch, pipeline in branch_pipelines.items()
        }

    def transform(self, results: dict) -> Optional[List[dict]]:
        """Transform function to apply transforms sequentially.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            list[dict]: Results from different pipeline.
        """
        multi_results = {}
        for branch, pipeline in self.branch_pipelines.items():
            branch_results = pipeline(copy.deepcopy(results))
            # If one branch pipeline returns None,
            # it will sample another data from dataset.
            if branch_results is None:
                return None
            multi_results[branch] = branch_results
        return multi_results

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
