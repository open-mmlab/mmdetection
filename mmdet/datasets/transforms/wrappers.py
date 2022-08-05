# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Callable, List, Optional, Union

from mmcv.transforms import BaseTransform, Compose
from mmcv.transforms.utils import cache_random_params

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
class ProposalBroadcaster(BaseTransform):
    """A transform wrapper to apply the wrapped transforms to process both
    `gt_bboxes` and `proposals` without adding any codes. It will do the
    following steps:

        1. Scatter the broadcasting targets to a list of inputs of the wrapped
           transforms. The type of the list should be list[dict, dict], which
           the first is the original inputs, the second is the processing
           results that `gt_bboxes` being rewritten by the `proposals`.
        2. Apply ``self.transforms``, with same random parameters, which is
           sharing with a context manager. The type of the outputs is a
           list[dict, dict].
        3. Gather the outputs, update the `proposals` in the first item of
           the outputs with the `gt_bboxes` in the second .

    Args:
         transforms (list[dict | callable]): Sequence of transform object or
            config dict to be wrapped.

    Examples:
        >>> pipeline = [
        >>>     dict(type='LoadImageFromFile'),
        >>>     dict(type='LoadProposals', num_max_proposals=2000),
        >>>     dict(type='LoadAnnotations', with_bbox=True),
        >>>     dict(
        >>>         type='ProposalBroadcaster',
        >>>         transforms=[
        >>>             dict(type='Resize', scale=(1333, 800),
        >>>                  keep_ratio=True),
        >>>             dict(type='RandomFlip', prob=0.5),
        >>>         ]),
        >>>     dict(type='PackDetInputs')]
    """

    def __init__(
        self,
        transforms: List[Union[dict, Callable[[dict], dict]]],
    ):
        if transforms is None:
            transforms = []
        self.transforms = Compose(transforms)

    def transform(self, results: dict) -> dict:
        """Apply wrapped transform functions to process both `gt_bboxes` and
        `proposals`.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        assert results.get('proposals', None) is not None, \
            '`proposals` should be in the results, please delete ' \
            '`ProposalBroadcaster` in your configs, or check whether ' \
            'you have load proposals successfully.'

        inputs = self._process_input(results)
        outputs = self._apply_transforms(inputs)
        outputs = self._process_output(outputs)
        return outputs

    def _process_input(self, data: dict) -> list:
        """Scatter the broadcasting targets to a list of inputs of the wrapped
        transforms.

        Args:
            data (dict): The original input data.

        Returns:
            list[dict, dict]: A list of input data.
        """
        cp_data = copy.deepcopy(data)
        cp_data['gt_bboxes'] = cp_data['proposals']
        scatters = [data, cp_data]
        return scatters

    def _apply_transforms(self, inputs: list) -> list:
        """Apply ``self.transforms``.

        Args:
            inputs (list[dict, dict]): list of input data.

        Returns:
            list[dict, dict]: The output of the wrapped pipeline.
        """
        ctx = cache_random_params
        with ctx(self.transforms):
            output_scatters = [self.transforms(_input) for _input in inputs]
        return output_scatters

    def _process_output(self, output_scatters: list) -> dict:
        """Gathering and renaming data items.

        Args:
            output_scatters (list[dict, dict]): The output of the wrapped
                pipeline.

        Returns:
            dict: Updated result dict.
        """
        assert isinstance(output_scatters, list) and \
               isinstance(output_scatters[0], dict) and \
               len(output_scatters) == 2
        outputs = output_scatters[0]
        outputs['proposals'] = output_scatters[1]['gt_bboxes']
        return outputs
