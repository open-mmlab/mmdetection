# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import mmengine
import sys
import types
import importlib.machinery
from mmengine.utils import digit_version

from .version import __version__, version_info

mmcv_minimum_version = '2.0.0rc4'
mmcv_maximum_version = '2.2.0'
mmcv_version = digit_version(mmcv.__version__)

mmengine_minimum_version = '0.7.1'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version < digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.'

assert (mmengine_version >= digit_version(mmengine_minimum_version)
        and mmengine_version < digit_version(mmengine_maximum_version)), \
    f'MMEngine=={mmengine.__version__} is used but incompatible. ' \
    f'Please install mmengine>={mmengine_minimum_version}, ' \
    f'<{mmengine_maximum_version}.'


try:
    import mmcv._ext  # noqa: F401
except ModuleNotFoundError:
    class _MissingMMCVExt(types.ModuleType):

        def __init__(self, name):
            super().__init__(name)
            self.__file__ = '<missing mmcv._ext>'
            self.__package__ = 'mmcv'
            self.__loader__ = None
            self.__spec__ = importlib.machinery.ModuleSpec(name, None)

        def __getattr__(self, name):
            if name.startswith('__') and name.endswith('__'):
                raise AttributeError(name)

            def _missing_op(*args, **kwargs):
                raise NotImplementedError(
                    f'MMCV compiled op "{name}" is unavailable. Install '
                    'full MMCV with ops for models that require it.')

            setattr(self, name, _missing_op)
            return _missing_op

    sys.modules.setdefault('mmcv._ext', _MissingMMCVExt('mmcv._ext'))

__all__ = ['__version__', 'version_info', 'digit_version']
