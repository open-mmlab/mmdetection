# from . import compiling_info
from .compiling_info import get_compiler_version, get_compiling_cuda_version

# get_compiler_version = compiling_info.get_compiler_version
# get_compiling_cuda_version = compiling_info.get_compiling_cuda_version

__all__ = ['get_compiler_version', 'get_compiling_cuda_version']
