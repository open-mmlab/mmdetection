# Copyright (c) OpenMMLab. All rights reserved.
import itertools

import numpy as np
import torch

from .general_data import GeneralData


class InstanceData(GeneralData):
    """Data structure for instance-level annnotations or predictions.

    Subclass of :class:`GeneralData`. All value in `data_fields`
    should have the same length. This design refer to
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/instances.py # noqa E501

    Examples:
        >>> from mmdet.core import InstanceData
        >>> import numpy as np
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> results = InstanceData(img_meta)
        >>> img_shape in results
        True
        >>> results.det_labels = torch.LongTensor([0, 1, 2, 3])
        >>> results["det_scores"] = torch.Tensor([0.01, 0.7, 0.6, 0.3])
        >>> results["det_masks"] = np.ndarray(4, 2, 2)
        >>> len(results)
        4
        >>> print(resutls)
        <InstanceData(

            META INFORMATION
        pad_shape: (800, 1216, 3)
        img_shape: (800, 1196, 3)

            PREDICTIONS
        shape of det_labels: torch.Size([4])
        shape of det_masks: (4, 2, 2)
        shape of det_scores: torch.Size([4])

        ) at 0x7fe26b5ca990>
        >>> sorted_results = results[results.det_scores.sort().indices]
        >>> sorted_results.det_scores
        tensor([0.0100, 0.3000, 0.6000, 0.7000])
        >>> sorted_results.det_labels
        tensor([0, 3, 2, 1])
        >>> print(results[results.scores > 0.5])
        <InstanceData(

            META INFORMATION
        pad_shape: (800, 1216, 3)
        img_shape: (800, 1196, 3)

            PREDICTIONS
        shape of det_labels: torch.Size([2])
        shape of det_masks: (2, 2, 2)
        shape of det_scores: torch.Size([2])

        ) at 0x7fe26b6d7790>
        >>> results[results.det_scores > 0.5].det_labels
        tensor([1, 2])
        >>> results[results.det_scores > 0.5].det_scores
        tensor([0.7000, 0.6000])
    """

    def __setattr__(self, name, value):

        if name in ('_meta_info_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f'{name} has been used as a '
                    f'private attribute, which is immutable. ')

        else:
            assert isinstance(value, (torch.Tensor, np.ndarray, list)), \
                f'Can set {type(value)}, only support' \
                f' {(torch.Tensor, np.ndarray, list)}'

            if self._data_fields:
                assert len(value) == len(self), f'the length of ' \
                                             f'values {len(value)} is ' \
                                             f'not consistent with' \
                                             f' the length ' \
                                             f'of this :obj:`InstanceData` ' \
                                             f'{len(self)} '
            super().__setattr__(name, value)

    def __getitem__(self, item):
        """
        Args:
            item (str, obj:`slice`,
                obj`torch.LongTensor`, obj:`torch.BoolTensor`):
                get the corresponding values according to item.

        Returns:
            obj:`InstanceData`: Corresponding values.
        """
        assert len(self), ' This is a empty instance'

        assert isinstance(
            item, (str, slice, int, torch.LongTensor, torch.BoolTensor))

        if isinstance(item, str):
            return getattr(self, item)

        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.new()
        if isinstance(item, (torch.Tensor)):
            assert item.dim() == 1, 'Only support to get the' \
                                 ' values along the first dimension.'
            if isinstance(item, torch.BoolTensor):
                assert len(item) == len(self), f'The shape of the' \
                                               f' input(BoolTensor)) ' \
                                               f'{len(item)} ' \
                                               f' does not match the shape ' \
                                               f'of the indexed tensor ' \
                                               f'in results_filed ' \
                                               f'{len(self)} at ' \
                                               f'first dimension. '

            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(v, list):
                    r_list = []
                    # convert to indexes from boolTensor
                    if isinstance(item, torch.BoolTensor):
                        indexes = torch.nonzero(item).view(-1)
                    else:
                        indexes = item
                    for index in indexes:
                        r_list.append(v[index])
                    new_data[k] = r_list
        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data

    @staticmethod
    def cat(instances_list):
        """Concat the predictions of all :obj:`InstanceData` in the list.

        Args:
            instances_list (list[:obj:`InstanceData`]): A list
                of :obj:`InstanceData`.

        Returns:
            obj:`InstanceData`
        """
        assert all(
            isinstance(results, InstanceData) for results in instances_list)
        assert len(instances_list) > 0
        if len(instances_list) == 1:
            return instances_list[0]

        new_data = instances_list[0].new()
        for k in instances_list[0]._data_fields:
            values = [results[k] for results in instances_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):
                values = np.concatenate(values, axis=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            else:
                raise ValueError(
                    f'Can not concat the {k} which is a {type(v0)}')
            new_data[k] = values
        return new_data

    def __len__(self):
        if len(self._data_fields):
            for v in self.values():
                return len(v)
        else:
            raise AssertionError('This is an empty `InstanceData`.')
