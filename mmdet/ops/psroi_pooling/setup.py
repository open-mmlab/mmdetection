from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='psroi_pooling_cuda',
    ext_modules=[
        CUDAExtension('psroi_pooling_cuda', [
            'src/psroi_pooling_cuda.cpp',
            'src/psroi_pooling_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
