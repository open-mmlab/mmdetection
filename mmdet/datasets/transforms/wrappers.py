# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from mmcv.transforms import BaseTransform, Compose
from mmcv.transforms.utils import cache_random_params, cache_randomness

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MultiBranch(BaseTransform):
    r"""Multiple branch pipeline wrapper.

    Generate multiple data-augmented versions of the same image.
    `MultiBranch` needs to specify the branch names of all
    pipelines of the dataset, perform corresponding data augmentation
    for the current branch, and return None for other branches,
    which ensures the consistency of return format across
    different samples.

    Args:
        branch_field (list): List of branch names.
        branch_pipelines (dict): Dict of different pipeline configs
            to be composed.

    Examples:
        >>> branch_field = ['sup', 'unsup_teacher', 'unsup_student']
        >>> sup_pipeline = [
        >>>     dict(type='LoadImageFromFile'),
        >>>     dict(type='LoadAnnotations', with_bbox=True),
        >>>     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        >>>     dict(type='RandomFlip', prob=0.5),
        >>>     dict(
        >>>         type='MultiBranch',
        >>>         branch_field=branch_field,
        >>>         sup=dict(type='PackDetInputs'))
        >>>     ]
        >>> weak_pipeline = [
        >>>     dict(type='LoadImageFromFile'),
        >>>     dict(type='LoadAnnotations', with_bbox=True),
        >>>     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        >>>     dict(type='RandomFlip', prob=0.0),
        >>>     dict(
        >>>         type='MultiBranch',
        >>>         branch_field=branch_field,
        >>>         sup=dict(type='PackDetInputs'))
        >>>     ]
        >>> strong_pipeline = [
        >>>     dict(type='LoadImageFromFile'),
        >>>     dict(type='LoadAnnotations', with_bbox=True),
        >>>     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        >>>     dict(type='RandomFlip', prob=1.0),
        >>>     dict(
        >>>         type='MultiBranch',
        >>>         branch_field=branch_field,
        >>>         sup=dict(type='PackDetInputs'))
        >>>     ]
        >>> unsup_pipeline = [
        >>>     dict(type='LoadImageFromFile'),
        >>>     dict(type='LoadEmptyAnnotations'),
        >>>     dict(
        >>>         type='MultiBranch',
        >>>         branch_field=branch_field,
        >>>         unsup_teacher=weak_pipeline,
        >>>         unsup_student=strong_pipeline)
        >>>     ]
        >>> from mmcv.transforms import Compose
        >>> sup_branch = Compose(sup_pipeline)
        >>> unsup_branch = Compose(unsup_pipeline)
        >>> print(sup_branch)
        >>> Compose(
        >>>     LoadImageFromFile(ignore_empty=False, to_float32=False, color_type='color', imdecode_backend='cv2') # noqa
        >>>     LoadAnnotations(with_bbox=True, with_label=True, with_mask=False, with_seg=False, poly2mask=True, imdecode_backend='cv2') # noqa
        >>>     Resize(scale=(1333, 800), scale_factor=None, keep_ratio=True, clip_object_border=True), backend=cv2), interpolation=bilinear) # noqa
        >>>     RandomFlip(prob=0.5, direction=horizontal)
        >>>     MultiBranch(branch_pipelines=['sup'])
        >>> )
        >>> print(unsup_branch)
        >>> Compose(
        >>>     LoadImageFromFile(ignore_empty=False, to_float32=False, color_type='color', imdecode_backend='cv2') # noqa
        >>>     LoadEmptyAnnotations(with_bbox=True, with_label=True, with_mask=False, with_seg=False, seg_ignore_label=255) # noqa
        >>>     MultiBranch(branch_pipelines=['unsup_teacher', 'unsup_student'])
        >>> )
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
         transforms (list, optional): Sequence of transform
            object or config dict to be wrapped. Defaults to [].

    Note: The `TransformBroadcaster` in MMCV can achieve the same operation as
          `ProposalBroadcaster`, but need to set more complex parameters.

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

    def __init__(self, transforms: List[Union[dict, Callable]] = []) -> None:
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
            list[dict]: A list of input data.
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
            list[dict]: The output of the wrapped pipeline.
        """
        assert len(inputs) == 2
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
