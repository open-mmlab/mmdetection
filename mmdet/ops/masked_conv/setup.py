from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='masked_conv2d_cuda',
    ext_modules=[
        CUDAExtension('masked_conv2d_cuda', [
            'src/masked_conv2d_cuda.cpp',
            'src/masked_conv2d_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
