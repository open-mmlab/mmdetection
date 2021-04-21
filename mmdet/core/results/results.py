import copy
import itertools

import numpy as np
import torch

from mmdet.core import bbox2result
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

    def __init__(self, img_meta=None, **kwargs):
        """
        Args:
            img_meta (dict): Meta information about the image.
            kwargs (dict): fields to add to this `Instances`.
        """

        self._meta_info_field = dict()
        self._results_field = dict()

        if img_meta is not None:
            img_meta = copy.deepcopy(img_meta)
            for k, v in img_meta.items():
                self._meta_info_field[k] = v

        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def new_results(self):
        """Return a new results with same image meta information and empty
        results_field."""
        new_results = self.__class__()
        for k, v in self.__dict__.items():
            if k != '_results_field':
                new_results[k] = copy.deepcopy(v)
            else:
                new_results[k] = dict()
        return new_results

    def export(self, ressults_type='bbox', num_classes=80):
        """Export results field to a dict, all tensor in results field would be
        converted to numpy."""
        assert ressults_type in ('bbox', 'mask')
        if ressults_type == 'bbox':
            det_bboxes = torch.cat([self.bboxes, self.scores[:, None]], -1)
            return bbox2result(det_bboxes, self.labels, num_classes)

    def keys(self):
        return list(self._results_field.keys()) + \
               list(self._meta_info_field.keys())

    def values(self):
        return list(self._results_field.values()) + \
               list(self._meta_info_field.values())

    def items(self):
        items = []
        for k, v in self._results_field.items():
            items.append((k, v))
        for k, v in self._meta_info_field.items():
            items.append((k, v))
        return items

    def __setattr__(self, name, val):
        if name.startswith('_'):
            super().__setattr__(name, val)
        elif name in self._meta_info_field:
            raise AttributeError(f'{name} is used in meta information,'
                                 f'which is unmodifiable')
        else:
            self.set(name, val)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        if name in self._meta_info_field:
            return self._meta_info_field[name]
        elif name in self._results_field:
            return self._results_field[name]
        elif name in dir(self):
            return super(Results, self).__getattr__(name)
        else:
            raise AttributeError(name)

    __getitem__ = __getattr__

    def set(self, name, value):
        if isinstance(value, torch.Tensor) and self.device:
            self._results_field[name] = value.to(self.device)
        else:
            self._results_field[name] = value

    def get(self, *args):
        assert len(args) < 3, '`get` get more than 2 arguments'
        name = args[0]
        if name in self._results_field:
            return self._results_field.get(name)
        elif name in self._meta_info_field:
            return self._meta_info_field.get(name)
        elif name.startswith('_'):
            return self.__dict__.get(*args)

    def pop(self, *args):
        assert len(args) < 3, '`pop` get more than 2 arguments'
        name = args[0]
        if name in self._results_field:
            return self._results_field.pop(name)
        elif name in self._meta_info_field:
            raise KeyError(f'{name} is a key in meta information, '
                           f'which is unmodifiable')
        elif name.startswith('_'):
            return self.__dict__.pop(*args)

    def results(self):
        return copy.deepcopy(self._results_field)

    def meta_info(self):
        return copy.deepcopy(self._meta_info_field)

    @property
    def device(self):
        """Return the device of all tensor in results field, return None when
        results field is empty."""
        device = None
        for v in self._results_field.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return device

    def __contains__(self, item):
        return item in self._results_field or \
                    item in self._meta_info_field

    def __delattr__(self, item):
        if not item.startswith('_'):
            if item in self._meta_info_field:
                raise KeyError(f'{item} is used in meta information, '
                               f'which is unmodifiable.')
            del self._results_field[item]
        elif item in ('_meta_info_field', '_results_field'):
            raise AssertionError(f'You can not delete {item}')
        else:
            super().__delattr__(item)

    __delitem__ = __delattr__

    # Tensor-like methods
    def to(self, *args, **kwargs):
        """Apply same name function to all tensors in results field."""
        new_instance = self.new_results()
        for k, v in self._results_field.items():
            if isinstance(v, torch.Tensor):
                new_instance[k] = v.to(*args, **kwargs)
        return new_instance

    # Tensor-like methods
    def cpu(self):
        """Apply same name function to all tensors in results field."""
        new_instance = self.new_results()
        for k, v in self._results_field.items():
            if isinstance(v, torch.Tensor):
                new_instance[k] = v.cpu()
        return new_instance

    # Tensor-like methods
    def cuda(self):
        """Apply same name function to all tensors in results field."""
        new_instance = self.new_results()
        for k, v in self._results_field.items():
            if isinstance(v, torch.Tensor):
                new_instance[k] = v.cuda()
        return new_instance

    # Tensor-like methods
    def detach(self):
        """Apply same name function to all tensors in results field."""
        new_instance = self.new_results()
        for k, v in self._results_field.items():
            if isinstance(v, torch.Tensor):
                new_instance[k] = v.detach()
        return new_instance

    # Tensor-like methods
    def numpy(self):
        """Apply same name function to all tensors in results field."""
        new_instance = self.new_results()
        for k, v in self._results_field.items():
            if isinstance(v, torch.Tensor):
                new_instance[k] = v.cpu().numpy()
        return new_instance

    def __nice__(self):
        repr = '\n   META INFORMATION \n'
        for k, v in self._meta_info_field.items():
            repr += f'{k}: {v} \n'
        repr += '\n   PREDICTIONS \n'
        for k, v in self._results_field.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                repr += f'shape of {k}: {v.shape} \n'
            else:
                repr += f'{k}: {v} \n'
        return repr

    def __copy__(self):
        # Due to the overloading of  `__getattr__`,
        # have to customize the `__copy__`.
        new_instance = self.__class__()
        new_instance.__dict__.update(self.__dict__)
        return new_instance

    def __deepcopy__(self, memo=None):
        # Due to the overloading of  `__getattr__`,
        # have to customize the `__deepcopy__`.
        new_instance = self.__class__()
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                new_instance[k] = copy.deepcopy(v)
        return new_instance


class InstanceResults(Results):

    def __len__(self):
        for v in self._results_field.values():
            return v.__len__()
        raise NotImplementedError('This is an empty instance.')

    def set(self, name, value):
        assert isinstance(value, (torch.Tensor, np.ndarray, list)), \
            f'Can set {type(value)}, only support' \
            f' {(torch.Tensor, np.ndarray, list)}'

        if isinstance(value, torch.Tensor) and self.device:
            value = value.to(self.device)

        for v in self._results_field.values():
            assert len(v) == len(value), f'the length of ' \
                                         f'values {len(value)} is ' \
                                         f'not consistent with the length ' \
                                         f'of this instancne {len(self)}'
        self._results_field[name] = value

    def __getitem__(self, item):
        """
        Args:
            item (str:`torch.LongTensor`, obj:`torch.BoolTensor`):
                get the corresponding values according to item.

        Returns:
            obj:`InstanceResults`: Corresponding values.
        """

        if isinstance(item, str):
            super().__getitem__(item)

        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError('Instances index out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        r_results = self.new_results()
        if isinstance(item, (torch.LongTensor, torch.BoolTensor)):
            assert item.dim() == 1, 'Only support to get the' \
                                 ' values along the first dimension.'
            for k, v in self._results_field.items():
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
            for k, v in self._results_field.items():
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
        assert all(isinstance(i, InstanceResults) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        cat_results = instance_lists[0].new_results()
        for k in instance_lists[0]._results_field.keys():
            values = [i.get(k) for i in instance_lists]
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
                    f'Can not concat the {k} which is a {type(k)}')
            cat_results[k] = values
        return cat_results


if __name__ == '__main__':
    r = Results(img_meta=dict(name='123'))
    print(r)
