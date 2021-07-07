import copy
import itertools
from itertools import chain

import numpy as np
import torch

from mmdet.utils.util_mixins import NiceRepr


class Results(NiceRepr):
    """A general data structure to store the model's results.

    The attributes of `Results` are divided into two parts,
    the `meta_info_field` and the `results_field` respectively.

    The `meta_info_field` usually includes the information about
    the image such as filename, image_shape, padding_shape, etc.
    All attributed in this filed is immutable once set,
    but the user can add new meta information with
    `add_meta_info` function.


    The model predictions are stored in `results_field`.
    Models predictions can be accessed
    or modified by dict-like or object-like operations
    such as  `.` , `[]`, `in`, `del`, `pop(str)` `get(str)`, `keys()`,
    `values()`, `items()`. User can also apply tensor-like methods
    to all obj:`torch.Tensor` in the `results_filed`,
    such as `.cuda()`, `.cpu()`, `.numpy()`, `device`, `.to()`
    `.detach()`, `.numpy()`

    Args:
        img_meta (dict): A dict contains the meta information
            of image. such as `img_shape`, `scale_factor`, etc.

    Examples:
        >>> from mmdet.core.results.results import Results
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> results = Results(img_meta)
        >>> img_shape in results
        True
        >>> results.det_labels = torch.LongTensor([0, 1, 2, 3])
        >>> results["det_scores"] = torch.Tensor([0.01, 0.1, 0.2, 0.3])
        >>> print(results)
        <Results(

          META INFORMATION
        img_shape: (800, 1196, 3)
        pad_shape: (800, 1216, 3)

           PREDICTIONS
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

    def __init__(self, meta=None):
        """
        Args:
            meta (dict): Meta information about the image.
        """

        self._meta_info_field = set()
        self._results_field = set()

        if meta is not None:
            self.add_meta_info(meta=meta)

    def add_meta_info(self, meta):
        assert isinstance(
            meta,
            dict), f'meta should be a `dict` but get {self.__class__.__name__}'
        meta = copy.deepcopy(meta)
        for k, v in meta.items():
            if k in self._meta_info_field:
                raise KeyError(
                    f'img_meta_info {k} has been set as '
                    f'{getattr(self, k)} before, which is immutable ')
            else:
                self._meta_info_field.add(k)
                self.__dict__[k] = v

    def new_results(self):
        """Return a new results with same image meta information and empty
        results_field."""
        new_results = self.__class__()
        new_results.add_meta_info(self.meta_info_field)
        return new_results

    def keys(self):
        for key in chain(self._results_field, self._meta_info_field):
            yield key

    def values(self):
        for k in self.keys():
            yield getattr(self, k)

    def items(self):
        for k in self.keys():
            yield (k, getattr(self, k))

    def __setattr__(self, name, val):
        if name in ('_meta_info_field', '_results_field'):
            if not hasattr(self, name):
                super().__setattr__(name, val)
            else:
                raise AttributeError(
                    f'{name} has been used as a '
                    f'private attribute, which is immutable. ')
        else:
            if name in self._meta_info_field:
                raise AttributeError(f'`{name}` is used in meta information,'
                                     f'which is immutable')

            self._results_field.add(name)
            super().__setattr__(name, val)

    def __delattr__(self, item):

        if item in ('_meta_info_field', '_results_field'):
            raise AttributeError(f'You can not delete {item}')

        if item in self._meta_info_field:
            raise KeyError(f'{item} is used in meta information, '
                           f'which is immutable.')
        super().__delattr__(item)
        if item in self._results_field:
            self._results_field.remove(item)

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
        if name in self._meta_info_field:
            raise KeyError(f'{name} is a key in meta information, '
                           f'which is immutable')

        if args[0] in self._results_field:
            self._results_field.remove(args[0])
            return self.__dict__.pop(*args)

        # with default value
        elif len(args) == 2:
            return args[1]
        else:
            raise KeyError(f'{args[0]}')

    @property
    def results_keys(self):
        return list(copy.copy(self._results_field))

    @property
    def meta_info_keys(self):
        return list(copy.copy(self._meta_info_field))

    @property
    def results_field(self):
        return {k: getattr(self, k) for k in self._results_field}

    @property
    def meta_info_field(self):
        return {
            k: copy.deepcopy(getattr(self, k))
            for k in self._meta_info_field
        }

    def __contains__(self, item):
        return item in self._results_field or \
                    item in self._meta_info_field

    # Tensor-like methods
    def to(self, *args, **kwargs):
        """Apply same name function to all tensors in results field."""
        new_results = self.new_results()
        for k, v in self.results_field.items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
            new_results[k] = v
        return new_results

    # Tensor-like methods
    def cpu(self):
        """Apply same name function to all tensors in results field."""
        new_results = self.new_results()
        for k, v in self.results_field.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu()
            new_results[k] = v
        return new_results

    # Tensor-like methods
    def cuda(self):
        """Apply same name function to all tensors in results field."""
        new_results = self.new_results()
        for k, v in self.results_field.items():
            if isinstance(v, torch.Tensor):
                v = v.cuda()
            new_results[k] = v
        return new_results

    # Tensor-like methods
    def detach(self):
        """Apply same name function to all tensors in results field."""
        new_results = self.new_results()
        for k, v in self.results_field.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            new_results[k] = v
        return new_results

    # Tensor-like methods
    def numpy(self):
        """Apply same name function to all tensors in results field."""
        new_results = self.new_results()
        for k, v in self.results_field.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            new_results[k] = v
        return new_results

    def __nice__(self):
        repr = '\n \n  META INFORMATION \n'
        for k, v in self.meta_info_field.items():
            repr += f'{k}: {v} \n'
        repr += '\n   PREDICTIONS \n'
        for k, v in self.results_field.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                repr += f'shape of {k}: {v.shape} \n'
            else:
                repr += f'{k}: {v} \n'
        return repr + '\n'


class InstanceResults(Results):
    """Subclass of results. All value in `results_field` should has same
    length.

    The code is modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/instances.py # noqa E501

    Examples:
        >>> from mmdet.core.results.results import InstanceResults
        >>> import numpy as np
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> results = InstanceResults(img_meta)
        >>> img_shape in results
        True
        >>> results.det_labels = torch.LongTensor([0, 1, 2, 3])
        >>> results["det_scores"] = torch.Tensor([0.01, 0.7, 0.6, 0.3])
        >>> results["det_masks"] = np.ndarray(4, 2, 2)
        >>> len(results)
        4
        >>> print(resutls)
        <Results(

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
        <InstanceResults(

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

        if name in ('_meta_info_field', '_results_field'):
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

            if self._results_field:
                assert len(value) == len(self), f'the length of ' \
                                             f'values {len(value)} is ' \
                                             f'not consistent with' \
                                             f' the length ' \
                                             f'of this instancne {len(self)}'
            super().__setattr__(name, value)

    def __getitem__(self, item):
        """
        Args:
            item (str, obj:`slice`,
                obj`torch.LongTensor`, obj:`torch.BoolTensor`):
                get the corresponding values according to item.

        Returns:
            obj:`InstanceResults`: Corresponding values.
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
                                               f'{len(item)} at index 0' \
                                               f' does not match the shape ' \
                                               f'of the indexed tensor ' \
                                               f'in results_filed ' \
                                               f'{len(self)} at index 0'

            for k, v in self.results_field.items():
                if isinstance(v, torch.Tensor):
                    r_results[k] = v[item]
                elif isinstance(v, np.ndarray):
                    r_results[k] = v[item.cpu().numpy()]
                elif isinstance(v, list):
                    r_list = []
                    if isinstance(item, torch.BoolTensor):
                        indexs = torch.nonzero(item).view(-1)
                    else:
                        indexs = item
                    for index in indexs:
                        r_list.append(v[index])
                    r_results[k] = r_list
        else:
            # item is a slice
            for k, v in self.results_field.items():
                r_results[k] = v[item]
        return r_results

    @staticmethod
    def cat(instance_lists):
        """Concat the predictions of all InstanceResults in the list.

        Args:
            instance_lists(list[InstanceResults]): A list of InstanceResults.

        Returns:
            obj:`InstanceResults`
        """
        assert all(
            isinstance(results, InstanceResults) for results in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        cat_results = instance_lists[0].new_results()
        for k in instance_lists[0]._results_field:
            values = [results[k] for results in instance_lists]
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
        if len(self._results_field):
            for v in self.results_field.values():
                return len(v)
        else:
            raise AssertionError('This is an empty `InstanceResults`.')
