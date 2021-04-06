import copy

import numpy as np
import torch


class Results(object):
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

    def export_results(self):
        """Export the predictions of the model."""
        return copy.deepcopy(self._results_field)

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
        else:
            raise AttributeError(f'Can not find {name}')

    def set(self, name, value):
        self._results_field[name] = value

    def has(self, name):
        return name in self._results_field or name in self._meta_info_field

    def remove(self, name):
        del self._results_field[name]

    def results(self):
        return copy.deepcopy(self._results_field)

    def meta_info(self):
        return copy.deepcopy(self._meta_info_field)

    # Tensor-like methods
    def to(self, *args, **kwargs):
        new_instance = copy.deepcopy(self)
        for k, v in new_instance.items():
            if isinstance(v, torch.Tensor):
                new_instance[k] = v.to(*args, **kwargs)
        return new_instance

    # Tensor-like methods
    def cpu(self):
        new_instance = self.__deepcopy__()
        for k, v in new_instance.items():
            if isinstance(v, torch.Tensor):
                new_instance[k] = v.cpu()
        return new_instance

    # Tensor-like methods
    def cuda(self):
        new_instance = self.__deepcopy__()
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                new_instance[k] = copy.deepcopy(v)
        for k, v in new_instance.items():
            if isinstance(v, torch.Tensor):
                new_instance[k] = v.cuda()
        return new_instance

    # Tensor-like methods
    def detach(self):
        new_instance = self.__deepcopy__()
        for k, v in new_instance.items():
            if isinstance(v, torch.Tensor):
                v.requires_grad_(False)
        return new_instance

    # Tensor-like methods
    def numpy(self):
        new_instance = self.__deepcopy__()
        for k, v in new_instance.items():
            if isinstance(v, torch.Tensor):
                new_instance[k] = v.numpy()
        return new_instance

    def __str__(self):
        repr = f'{self.__class__.__name__}: \n'
        repr += '\n   META INFORMATION \n'
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

    __repr__ = __str__
