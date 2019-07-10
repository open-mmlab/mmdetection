from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {"cxx": [],
                      "nvcc": [
                          "-D__CUDA_NO_HALF_OPERATORS__",
                          "-D__CUDA_NO_HALF_CONVERSIONS__",
                          "-D__CUDA_NO_HALF2_OPERATORS__",
                      ]}

setup(
    name='roi_pool',
    ext_modules=[
        CUDAExtension('roi_pool_cuda',
                      ['src/roi_pool_cuda.cpp', 'src/roi_pool_kernel.cu', ],
                      extra_compile_args=extra_compile_args)
    ],
    cmdclass={'build_ext': BuildExtension})
