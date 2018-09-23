from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        content = f.read()
    return content


def get_version():
    version_file = 'mmdet/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


setup(
    name='mmdet',
    version=get_version(),
    description='Open MMLab Detection Toolbox',
    long_description=readme(),
    keywords='computer vision, object detection',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities',
    ],
    license='GPLv3',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=['numpy', 'matplotlib', 'six', 'terminaltables'],
    zip_safe=False)
