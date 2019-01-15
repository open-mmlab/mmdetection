from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='deform_conv_cuda',
    ext_modules=[
        CUDAExtension('deform_conv_cuda', [
            'src/deform_conv_cuda.cpp',
            'src/deform_conv_cuda_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
