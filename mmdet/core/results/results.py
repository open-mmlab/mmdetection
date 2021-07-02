import copy
import itertools

import numpy as np
import torch


from itertools import chain
from mmdet.utils.util_mixins import NiceRepr


class Results(NiceRepr):
    """Base class of the model's results.

    The model predictions and meta information are stored in this class.
    User can modify the predictions with `set()` or `remove()`.
    All predictions results can be export to a dict with `export_results`.

    The meta information(filename,image_size...) about the image are
    unmodifiable once initialized.

    The keys and values of meta information and predictions
    can be accessed with `keys()`, `values()`, `items()`.

    All other (non-field) attributes of this class are considered private:
    they must start with '_'.

    Besides, it support tensor like methods, such as `to()`,`cpu()`,
    `cuda()`,`numpy()`, which would apply corresponding function
    to all Tensors in the instance.
    """

    def __init__(self, img_meta=None):
        """
        Args:
            img_meta (dict): Meta information about the image.
        """

        self._meta_info_field = set()
        self._results_field = set()

        if img_meta is not None:
            self.add_meta_info(img_meta=img_meta)

    def add_meta_info(self, img_meta):
        assert isinstance(img_meta, dict), f"img_meta should be a `dict` but get {self.__class__.__name__}"
        img_meta = copy.deepcopy(img_meta)
        for k, v in img_meta.items():
            if k in self._meta_info_field:
                raise RuntimeError(f"img_meta_info {k} has been set as {getattr(self, k)} before, which is unmodifiable ")
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
        return chain(self._results_field,
               self._meta_info_field)

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
            if name in self._meta_info_field:
                raise AttributeError(f'`{name}` is used in meta information,'
                                     f'which is unmodifiable')
            if isinstance(val, torch.Tensor):
                if self.device:
                    val = val.to(self.device)
            self._results_field.add(name)
            super().__setattr__(name, val)


    # dict-like methods
    __setitem__ = __setattr__

    def __getitem__(self, name):
        return getattr(self, name)


    def get(self, *args):
        assert len(args) < 3, '`pop` get more than 2 arguments'
        return self.__dict__.get(*args)

    def pop(self, *args):
        assert len(args) < 3, '`pop` get more than 2 arguments'
        name = args[0]
        if name in self._meta_info_field:
            raise KeyError(f'{name} is a key in meta information, '
                           f'which is unmodifiable')

        if args[0] in self._results_field:
            self._results_field.remove(args[0])
            return self.__dict__.pop(*args)

        # with default value
        elif len(args) == 2:
            return args[1]
        else:
            raise KeyError(f"{args[0]}")

    @property
    def results_field(self):
        return {k: getattr(self,k) for k in self._results_field}

    @property
    def meta_info_field(self):
        return {k: copy.deepcopy(getattr(self, k)) for k in self._meta_info_field}

    @property
    def device(self):
        """Return the device of all tensor in results field, return None when
        results field is empty."""
        device = None
        for k in self._results_field:
            if isinstance(getattr(self, k), torch.Tensor):
                return  getattr(self, k).device
        return device

    def __contains__(self, item):
        return item in self._results_field or \
                    item in self._meta_info_field

    def __delattr__(self, item):

        if item in ('_meta_info_field', '_results_field'):
            raise AssertionError(f'You can not delete {item}')

        if item in self._meta_info_field:
            raise KeyError(f'{item} is used in meta information, '
                           f'which is unmodifiable.')
        super().__delattr__(item)
        if item in self._results_field:
            self._results_field.remove(item)

    __delitem__ = __delattr__

    # Tensor-like methods
    def to(self, *args, **kwargs):
        """Apply same name function to all tensors in results field."""
        new_results = self.new_results()
        for k, v in self.results_field.items():
            if isinstance(v, torch.Tensor):
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


    def __setattr__(self, name, value):

        if name in ('_meta_info_field', '_results_field'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
        else:
            assert isinstance(value, (torch.Tensor, np.ndarray, list)), \
                f'Can set {type(value)}, only support' \
                f' {(torch.Tensor, np.ndarray, list)}'

            if isinstance(value, torch.Tensor) and self.device:
                value = value.to(self.device)

            for v in self.results_field.values():
                assert len(v) == len(value), f'the length of ' \
                                             f'values {len(value)} is ' \
                                             f'not consistent with the length ' \
                                             f'of this instancne {len(self)}'
            super().__setattr__(name, value)

    def __getitem__(self, item):
        """
        Args:
            item (str, obj:`slice` ,obj`torch.LongTensor`, obj:`torch.BoolTensor`):
                get the corresponding values according to item.

        Returns:
            obj:`InstanceResults`: Corresponding values.
        """

        assert isinstance(item, (str,slice ,int ,torch.LongTensor, torch.BoolTensor))

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
        assert all(isinstance(results, InstanceResults) for results in instance_lists)
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
        for v in self.results_field.values():
            return len(v)
        raise NotImplementedError('This is an empty `InstanceResults`.')
