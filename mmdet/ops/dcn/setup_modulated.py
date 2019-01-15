from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='modulated_deform_conv',
    ext_modules=[
        CUDAExtension('modulated_dcn', [
            'src/modulated_dcn_cuda.cpp',
            'src/modulated_deform_im2col_cuda.cu',
            'src/deform_psroi_pooling_cuda.cu'
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
