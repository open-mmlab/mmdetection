# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import subprocess
import sys

import pytorch_sphinx_theme

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'MMDetection'
copyright = '2018-2021, OpenMMLab'
author = 'MMDetection Authors'
version_file = '../../mmdet/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


# The full version, including alpha/beta/rc tags
release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'recommonmark',
    'sphinx_markdown_tables',
    'sphinx_copybutton',
]

autodoc_mock_imports = [
    'matplotlib', 'pycocotools', 'terminaltables', 'mmdet.version', 'mmcv.ops'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

html_theme_options = {
    'menu': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/open-mmlab/mmdetection'
        },
        {
            'name':
            '算法库',
            'children': [{
                'name': 'MMCV',
                'url': 'https://mmcv.readthedocs.io/zh_CN/latest/',
                'description': '计算机视觉基础库'
            }, {
                'name': 'MMDetection',
                'url': 'https://mmdetection.readthedocs.io/zh_CN/latest/',
                'description': '检测工具箱与测试基准'
            }, {
                'name': 'MMAction2',
                'url': 'https://mmaction2.readthedocs.io/zh_CN/latest/',
                'description': '视频理解工具箱与测试基准'
            }, {
                'name': 'MMClassification',
                'url': 'https://mmclassification.readthedocs.io/zh_CN/latest/',
                'description': '图像分类工具箱与测试基准'
            }, {
                'name': 'MMSegmentation',
                'url': 'https://mmsegmentation.readthedocs.io/zh_CN/latest/',
                'description': '语义分割工具箱与测试基准'
            }, {
                'name': 'MMDetection3D',
                'url': 'https://mmdetection3d.readthedocs.io/zh_CN/latest/',
                'description': '通用3D目标检测平台'
            }, {
                'name': 'MMEditing',
                'url': 'https://mmediting.readthedocs.io/zh_CN/latest/',
                'description': '图像视频编辑工具箱'
            }, {
                'name': 'MMOCR',
                'url': 'https://mmocr.readthedocs.io/zh_CN/latest/',
                'description': '全流程文字检测识别理解工具包'
            }, {
                'name': 'MMPose',
                'url': 'https://mmpose.readthedocs.io/zh_CN/latest/',
                'description': '姿态估计工具箱与测试基准'
            }, {
                'name': 'MMTracking',
                'url': 'https://mmtracking.readthedocs.io/zh_CN/latest/',
                'description': '一体化视频目标感知平台'
            }, {
                'name': 'MMGeneration',
                'url': 'https://mmgeneration.readthedocs.io/zh_CN/latest/',
                'description': '生成模型工具箱'
            }, {
                'name': 'MMFlow',
                'url': 'https://mmflow.readthedocs.io/zh_CN/latest/',
                'description': '光流估计工具箱与测试基准'
            }, {
                'name': 'MMFewShot',
                'url': 'https://mmfewshot.readthedocs.io/zh_CN/latest/',
                'description': '少样本学习工具箱与测试基准'
            }, {
                'name': 'MMHuman3D',
                'url': 'https://mmhuman3d.readthedocs.io/en/latest/',
                'description': 'OpenMMLab 人体参数化模型工具箱与测试基准.'
            }]
        },
        {
            'name':
            'OpenMMLab',
            'children': [
                {
                    'name': '官网',
                    'url': 'https://openmmlab.com/'
                },
                {
                    'name': 'GitHub',
                    'url': 'https://github.com/open-mmlab/'
                },
                {
                    'name': '推特',
                    'url': 'https://twitter.com/OpenMMLab'
                },
                {
                    'name': '知乎',
                    'url': 'https://zhihu.com/people/openmmlab'
                },
            ]
        },
    ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']

language = 'zh_CN'

# -- Extension configuration -------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True


def builder_inited_handler(app):
    subprocess.run(['./stat.py'])


def setup(app):
    app.connect('builder-inited', builder_inited_handler)
