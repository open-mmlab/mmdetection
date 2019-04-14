from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='SigmoidFocalLoss',
    ext_modules=[
        CUDAExtension('sigmoid_focal_loss_cuda', [
            'src/sigmoid_focal_loss.cpp',
            'src/sigmoid_focal_loss_cuda.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
