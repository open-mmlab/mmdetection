from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {"cxx": [],
                      "nvcc": [
                          "-D__CUDA_NO_HALF_OPERATORS__",
                          "-D__CUDA_NO_HALF_CONVERSIONS__",
                          "-D__CUDA_NO_HALF2_OPERATORS__",
                      ]}

setup(
    name='roi_align_cuda',
    ext_modules=[
        CUDAExtension('roi_align_cuda',
                      ['src/roi_align_cuda.cpp', 'src/roi_align_kernel.cu', ],
                      extra_compile_args=extra_compile_args),
    ],
    cmdclass={'build_ext': BuildExtension})
