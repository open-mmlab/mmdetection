import os
import platform
import subprocess
import time
from setuptools import Extension, find_packages, setup

import numpy as np
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


MAJOR = 0
MINOR = 6
PATCH = 0
SUFFIX = ''
SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)

version_file = 'mmdet/version.py'


def get_git_hash():

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    elif os.path.exists(version_file):
        try:
            from mmdet.version import __version__
            sha = __version__.split('+')[-1]
        except ImportError:
            raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'

    return sha


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}

__version__ = '{}'
short_version = '{}'
"""
    sha = get_hash()
    VERSION = SHORT_VERSION + '+' + sha

    with open(version_file, 'w') as f:
        f.write(content.format(time.asctime(), VERSION, SHORT_VERSION))


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def make_cuda_ext(name, module, sources):

    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings']
        }

    extension = Extension(
        '{}.{}'.format(module, name),
        [os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[np.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension


if __name__ == '__main__':
    write_version_py()
    setup(
        name='mmdet',
        version=get_version(),
        description='Open MMLab Detection Toolbox',
        long_description=readme(),
        keywords='computer vision, object detection',
        url='https://github.com/open-mmlab/mmdetection',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        package_data={'mmdet.ops': ['*/*.so']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        license='Apache License 2.0',
        setup_requires=['pytest-runner', 'cython', 'numpy'],
        tests_require=['pytest'],
        install_requires=[
            'mmcv>=0.2.6', 'numpy', 'matplotlib', 'six', 'terminaltables',
            'pycocotools', 'torch>=1.1'
        ],
        ext_modules=[
            make_cython_ext(
                name='soft_nms_cpu',
                module='mmdet.ops.nms',
                sources=['src/soft_nms_cpu.pyx']),
            make_cuda_ext(
                name='roi_align_cuda',
                module='mmdet.ops.roi_align',
                sources=['src/roi_align_cuda.cpp', 'src/roi_align_kernel.cu']),
            make_cuda_ext(
                name='roi_pool_cuda',
                module='mmdet.ops.roi_pool',
                sources=['src/roi_pool_cuda.cpp', 'src/roi_pool_kernel.cu']),
            make_cuda_ext(
                name='nms_cpu',
                module='mmdet.ops.nms',
                sources=['src/nms_cpu.cpp']),
            make_cuda_ext(
                name='nms_cuda',
                module='mmdet.ops.nms',
                sources=['src/nms_cuda.cpp', 'src/nms_kernel.cu']),
            make_cuda_ext(
                name='deform_conv_cuda',
                module='mmdet.ops.dcn',
                sources=[
                    'src/deform_conv_cuda.cpp',
                    'src/deform_conv_cuda_kernel.cu'
                ]),
            make_cuda_ext(
                name='deform_pool_cuda',
                module='mmdet.ops.dcn',
                sources=[
                    'src/deform_pool_cuda.cpp',
                    'src/deform_pool_cuda_kernel.cu'
                ]),
            make_cuda_ext(
                name='sigmoid_focal_loss_cuda',
                module='mmdet.ops.sigmoid_focal_loss',
                sources=[
                    'src/sigmoid_focal_loss.cpp',
                    'src/sigmoid_focal_loss_cuda.cu'
                ]),
            make_cuda_ext(
                name='masked_conv2d_cuda',
                module='mmdet.ops.masked_conv',
                sources=[
                    'src/masked_conv2d_cuda.cpp', 'src/masked_conv2d_kernel.cu'
                ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
