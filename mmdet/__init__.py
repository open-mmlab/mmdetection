import mmcv

from .version import __version__, short_version

mmcv_version = '1.0.4'

assert mmcv.__version__ == mmcv_version, \
    f'MMCV=={mmcv.__version__} is used but incompatible.' \
    f'Please install mmcv=={mmcv_version}'

__all__ = ['__version__', 'short_version']
