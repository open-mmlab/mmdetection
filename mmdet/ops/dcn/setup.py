from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {"cxx": [],
                      "nvcc": [
                          "-D__CUDA_NO_HALF_OPERATORS__",
                          "-D__CUDA_NO_HALF_CONVERSIONS__",
                          "-D__CUDA_NO_HALF2_OPERATORS__",
                      ]}

setup(
    name='deform_conv',
    ext_modules=[
        CUDAExtension('deform_conv_cuda',
                      ['src/deform_conv_cuda.cpp', 'src/deform_conv_cuda_kernel.cu', ],
                      extra_compile_args=extra_compile_args),
    ],
    cmdclass={'build_ext': BuildExtension})

setup(
    name='deform_pool',
    ext_modules=[
        CUDAExtension('deform_pool_cuda',
                      ['src/deform_pool_cuda.cpp', 'src/deform_pool_cuda_kernel.cu'],
                      extra_compile_args=extra_compile_args),
    ],
    cmdclass={'build_ext': BuildExtension})
