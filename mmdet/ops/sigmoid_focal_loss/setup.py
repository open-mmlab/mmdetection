from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {"cxx": [],
                      "nvcc": [
                          "-D__CUDA_NO_HALF_OPERATORS__",
                          "-D__CUDA_NO_HALF_CONVERSIONS__",
                          "-D__CUDA_NO_HALF2_OPERATORS__",
                      ]}

setup(
    name='SigmoidFocalLoss',
    ext_modules=[
        CUDAExtension('sigmoid_focal_loss_cuda',
                      ['src/sigmoid_focal_loss.cpp', 'src/sigmoid_focal_loss_cuda.cu', ],
                      extra_compile_args=extra_compile_args),
    ],
    cmdclass={'build_ext': BuildExtension})
