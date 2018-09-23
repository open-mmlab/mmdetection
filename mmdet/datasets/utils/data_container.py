import functools

import torch


def assert_tensor_type(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError('{} has no attribute {} for type {}'.format(
                args[0].__class__.__name__, func.__name__, args[0].datatype))
        return func(*args, **kwargs)

    return wrapper


class DataContainer(object):

    def __init__(self, data, stack=False, padding_value=0, cpu_only=False):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self.data))

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()
