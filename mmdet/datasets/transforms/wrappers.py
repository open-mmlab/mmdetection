# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional

from mmcv.transforms import BaseTransform, Compose

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
