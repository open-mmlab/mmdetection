# Copyright (c) OpenMMLab. All rights reserved.
import copy
import itertools

import numpy as np
import torch

from mmdet.utils.util_mixins import NiceRepr


class GeneralData(NiceRepr):
    """A general data structure of OpenMMlab.

    A data structure that stores the meta information,
    the annotations of the images or the model predictions,
    which can be used in communication between components.

    The attributes in `GeneralData` are divided into two parts,
    the `meta_info_fields` and the `data_fields` respectively.

        - `meta_info_fields`: Usually contains the
          information about the image such as filename,
          image_shape, padding_shape, etc. All attributes in
          it are immutable once set,
          but the user can add new meta information with
          `set_meta` function, all information can be accessed
          with methods `meta_keys`, `meta_values`, `meta_items`.

        - `data_fields`: Annotations or model predictions are
          stored. The attributes can be accessed or modified by
          dict-like or object-like operations, such as
          `.` , `[]`, `in`, `del`, `pop(str)` `get(str)`, `keys()`,
          `values()`, `items()`. Users can also apply tensor-like methods
          to all obj:`torch.Tensor` in the `data_fileds`,
          such as `.cuda()`, `.cpu()`, `.numpy()`, `device`, `.to()`
          `.detach()`, `.numpy()`

    Args:
        meta_info (dict): A dict contains the meta information
            of single image. such as `img_shape`, `scale_factor`, etc.
            Default: None.
        data (dict): A dict contains annotations of single image or
            model predictions. Default: None.

    Examples:
        >>> from mmdet.core import GeneralData
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> results = GeneralData(meta_info=img_meta)
        >>> img_shape in results
        True
        >>> results.det_labels = torch.LongTensor([0, 1, 2, 3])
        >>> results["det_scores"] = torch.Tensor([0.01, 0.1, 0.2, 0.3])
        >>> print(results)
        <GeneralData(

          META INFORMATION
        img_shape: (800, 1196, 3)
        pad_shape: (800, 1216, 3)

          DATA FIELDS
        shape of det_labels: torch.Size([4])
        shape of det_scores: torch.Size([4])

        ) at 0x7f84acd10f90>
        >>> resutls.det_scores
        tensor([0.0100, 0.1000, 0.2000, 0.3000])
        >>> results.det_labels
        tensor([0, 1, 2, 3])
        >>> results['det_labels']
        tensor([0, 1, 2, 3])
        >>> 'det_labels' in results
        True
        >>> results.img_shape
        (800, 1196, 3)
        >>> 'det_scores' in results
        True
        >>> del results.det_scores
        >>> 'det_scores' in results
        False
        >>> det_labels = results.pop('det_labels', None)
        >>> det_labels
        tensor([0, 1, 2, 3])
        >>> 'det_labels' in results
        >>> False
    """

    def __init__(self, meta_info=None, data=None):

        self._meta_info_fields = set()
        self._data_fields = set()

        if meta_info is not None:
            self.set_meta_info(meta_info=meta_info)
        if data is not None:
            self.set_data(data)

    def set_meta_info(self, meta_info=None):
        """Add meta information.

        Args:
            meta_info (dict): A dict contains the meta information
                of image. such as `img_shape`, `scale_factor`, etc.
                Default: None.
        """
        assert isinstance(meta_info,
                          dict), f'meta should be a `dict` but get {meta_info}'
        meta = copy.deepcopy(meta_info)
        for k, v in meta.items():
            # should be consistent with original meta_info
            if k in self._meta_info_fields:
                ori_value = getattr(self, k)
                if isinstance(ori_value, (torch.Tensor, np.ndarray)):
                    if (ori_value == v).all():
                        continue
                    else:
                        raise KeyError(
                            f'img_meta_info {k} has been set as '
                            f'{getattr(self, k)} before, which is immutable ')
                elif ori_value == v:
                    continue
                else:
                    raise KeyError(
                        f'img_meta_info {k} has been set as '
                        f'{getattr(self, k)} before, which is immutable ')
            else:
                self._meta_info_fields.add(k)
                self.__dict__[k] = v

    def set_data(self, data=None):
        """Update a dict to `data_fields`.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions. Default: None.
        """
        assert isinstance(data,
                          dict), f'meta should be a `dict` but get {data}'
        for k, v in data.items():
            self.__setattr__(k, v)

    def new_results(self, meta_info=None, data=None):
        """Return a new results with same image meta information.

        Args:
            meta_info (dict): A dict contains the meta information
                of image. such as `img_shape`, `scale_factor`, etc.
                Default: None.
            data (dict): A dict contains annotations of image or
                model predictions. Default: None.
        """
        new_results = self.__class__()
        new_results.set_meta_info(dict(self.meta_items()))
        if meta_info is not None:
            new_results.set_meta_info(meta_info)
        if data is not None:
            new_results.set_data(data)
        return new_results

    def keys(self):
        """
        Returns:
            list: Contains all keys in data_fields.
        """
        return [key for key in self._data_fields]

    def meta_keys(self):
        """
        Returns:
            list: Contains all keys in meta_info_fields.
        """
        return [key for key in self._meta_info_fields]

    def values(self):
        """
        Returns:
            list: Contains all values in data_fields.
        """
        return [getattr(self, k) for k in self.keys()]

    def meta_values(self):
        """
        Returns:
            list: Contains all values in data_fields.
        """
        return [getattr(self, k) for k in self.meta_keys()]

    def items(self):
        for k in self.keys():
            yield (k, getattr(self, k))

    def meta_items(self):
        for k in self.meta_keys():
            yield (k, getattr(self, k))

    def __setattr__(self, name, val):
        if name in ('_meta_info_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, val)
            else:
                raise AttributeError(
                    f'{name} has been used as a '
                    f'private attribute, which is immutable. ')
        else:
            if name in self._meta_info_fields:
                raise AttributeError(f'`{name}` is used in meta information,'
                                     f'which is immutable')

            self._data_fields.add(name)
            super().__setattr__(name, val)

    def __delattr__(self, item):

        if item in ('_meta_info_fields', '_data_fields'):
            raise AttributeError(f'{item} has been used as a '
                                 f'private attribute, which is immutable. ')

        if item in self._meta_info_fields:
            raise KeyError(f'{item} is used in meta information, '
                           f'which is immutable.')
        super().__delattr__(item)
        if item in self._data_fields:
            self._data_fields.remove(item)

    # dict-like methods
    __setitem__ = __setattr__
    __delitem__ = __delattr__

    def __getitem__(self, name):
        return getattr(self, name)

    def get(self, *args):
        assert len(args) < 3, '`get` get more than 2 arguments'
        return self.__dict__.get(*args)

    def pop(self, *args):
        assert len(args) < 3, '`pop` get more than 2 arguments'
        name = args[0]
        if name in self._meta_info_fields:
            raise KeyError(f'{name} is a key in meta information, '
                           f'which is immutable')

        if args[0] in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)

        # with default value
        elif len(args) == 2:
            return args[1]
        else:
            raise KeyError(f'{args[0]}')

    def __contains__(self, item):
        return item in self._data_fields or \
                    item in self._meta_info_fields

    # Tensor-like methods
    def to(self, *args, **kwargs):
        """Apply same name function to all tensors in results field."""
        new_results = self.new_results()
        for k, v in self.items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
            new_results[k] = v
        return new_results

    # Tensor-like methods
    def cpu(self):
        """Apply same name function to all tensors in results field."""
        new_results = self.new_results()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu()
            new_results[k] = v
        return new_results

    # Tensor-like methods
    def cuda(self):
        """Apply same name function to all tensors in results field."""
        new_results = self.new_results()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                v = v.cuda()
            new_results[k] = v
        return new_results

    # Tensor-like methods
    def detach(self):
        """Apply same name function to all tensors in results field."""
        new_results = self.new_results()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            new_results[k] = v
        return new_results

    # Tensor-like methods
    def numpy(self):
        """Apply same name function to all tensors in results field."""
        new_results = self.new_results()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            new_results[k] = v
        return new_results

    def __nice__(self):
        repr = '\n \n  META INFORMATION \n'
        for k, v in self.meta_items():
            repr += f'{k}: {v} \n'
        repr += '\n   DATA FIELDS \n'
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                repr += f'shape of {k}: {v.shape} \n'
            else:
                repr += f'{k}: {v} \n'
        return repr + '\n'


class InstanceData(GeneralData):
    """Subclass of :class:`GeneralData`.

    All value in `data_fields` should has the same length.

    The code is modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/instances.py # noqa E501

    Examples:
        >>> from mmdet.core import InstanceData
        >>> import numpy as np
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> results = Instances(img_meta)
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
                                             f'of this :obj:`Instances` ' \
                                             f'{len(self)} '
            super().__setattr__(name, value)

    def __getitem__(self, item):
        """
        Args:
            item (str, obj:`slice`,
                obj`torch.LongTensor`, obj:`torch.BoolTensor`):
                get the corresponding values according to item.

        Returns:
            obj:`Instances`: Corresponding values.
        """
        assert len(self), ' This is a empty instance'

        assert isinstance(
            item, (str, slice, int, torch.LongTensor, torch.BoolTensor))

        if isinstance(item, str):
            return getattr(self, item)

        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError('Instances index out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        r_results = self.new_results()
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
                    r_results[k] = v[item]
                elif isinstance(v, np.ndarray):
                    r_results[k] = v[item.cpu().numpy()]
                elif isinstance(v, list):
                    r_list = []
                    # convert to indexes from boolTensor
                    if isinstance(item, torch.BoolTensor):
                        indexes = torch.nonzero(item).view(-1)
                    else:
                        indexes = item
                    for index in indexes:
                        r_list.append(v[index])
                    r_results[k] = r_list
        else:
            # item is a slice
            for k, v in self.items():
                r_results[k] = v[item]
        return r_results

    @staticmethod
    def cat(instances_list):
        """Concat the predictions of all :obj:`Instances` in the list.

        Args:
            instances_list(list[:obj:`Instances`]): A list of :obj:`Instances`.

        Returns:
            obj:`InstanceResults`
        """
        assert all(
            isinstance(results, InstanceData) for results in instances_list)
        assert len(instances_list) > 0
        if len(instances_list) == 1:
            return instances_list[0]

        cat_results = instances_list[0].new_results()
        for k in instances_list[0]._data_fields:
            values = [results[k] for results in instances_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):
                values = np.concatenate(values, axis=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), 'cat'):
                values = type(v0).cat(values)
            else:
                raise ValueError(
                    f'Can not concat the {k} which is a {type(v0)}')
            cat_results[k] = values
        return cat_results

    def __len__(self):
        if len(self._data_fields):
            for v in self.values():
                return len(v)
        else:
            raise AssertionError('This is an empty `Instances`.')
