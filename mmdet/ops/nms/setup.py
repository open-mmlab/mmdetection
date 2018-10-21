import os.path as osp
from distutils.core import setup, Extension

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# extensions
ext_args = dict(
    include_dirs=[np.get_include()],
    language='c++',
    extra_compile_args={
        'cc': ['-Wno-unused-function', '-Wno-write-strings'],
        'nvcc': ['-c', '--compiler-options', '-fPIC'],
    },
)

extensions = [
    Extension('cpu_nms', ['cpu_nms.pyx'], **ext_args),
    Extension('cpu_soft_nms', ['cpu_soft_nms.pyx'], **ext_args),
    Extension('gpu_nms', ['gpu_nms.pyx', 'nms_kernel.cu'], **ext_args),
]


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to cc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if osp.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', 'nvcc')
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['cc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):

    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


setup(
    name='nms',
    cmdclass={'build_ext': custom_build_ext},
    ext_modules=cythonize(extensions),
)
