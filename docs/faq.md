We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an issue using the [provided templates](https://github.com/open-mmlab/mmdetection/blob/master/.github/ISSUE_TEMPLATE/error-report.md) and make sure you fill in all required information in the template.

## MMCV Installation

- Compatibility issue between MMCV and MMDetection; "ConvWS is already registered in conv layer"; "AssertionError: MMCV==xxx is used but incompatible. Please install mmcv>=xxx, <=xxx."

  Please install the correct version of MMCV for the version of your MMDetection following the [installation instruction](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation).

- "No module named 'mmcv.ops'"; "No module named 'mmcv._ext'".

    1. Uninstall existing mmcv in the environment using `pip uninstall mmcv`.
    2. Install mmcv-full following the [installation instruction](https://mmcv.readthedocs.io/en/latest/#installation).

## PyTorch/CUDA Environment

- "invalid device function" or "no kernel image is available for execution".

    1. Check if your cuda runtime version (under `/usr/local/`), `nvcc --version` and `conda list cudatoolkit` version match.
    2. Run `python mmdet/utils/collect_env.py` to check whether PyTorch, torchvision, and MMCV are built for the correct GPU architecture.
    You may need to set `TORCH_CUDA_ARCH_LIST` to reinstall MMCV.
    The GPU arch table could be found [here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list),
    i.e. run `TORCH_CUDA_ARCH_LIST=7.0 pip install mmcv-full` to build MMCV for Volta GPUs.
    The compatibility issue could happen when using old GPUS, e.g., Tesla K80 (3.7) on colab.
    3. Check whether the running environment is the same as that when mmcv/mmdet has compiled.
    For example, you may compile mmcv using CUDA 10.0 but run it on CUDA 9.0 environments.

- "undefined symbol" or "cannot open xxx.so".

    1. If those symbols are CUDA/C++ symbols (e.g., libcudart.so or GLIBCXX), check whether the CUDA/GCC runtimes are the same as those used for compiling mmcv,
    i.e. run `python mmdet/utils/collect_env.py` to see if `"MMCV Compiler"`/`"MMCV CUDA Compiler"` is the same as `"GCC"`/`"CUDA_HOME"`.
    2. If those symbols are PyTorch symbols (e.g., symbols containing caffe, aten, and TH), check whether the PyTorch version is the same as that used for compiling mmcv.
    3. Run `python mmdet/utils/collect_env.py` to check whether PyTorch, torchvision, and MMCV are built by and running on the same environment.

- setuptools.sandbox.UnpickleableException: DistutilsSetupError("each element of 'ext_modules' option must be an Extension instance or 2-tuple")

    1. If you are using miniconda rather than anaconda, check whether Cython is installed as indicated in [#3379](https://github.com/open-mmlab/mmdetection/issues/3379).
    You need to manually install Cython first and then run command `pip install -r requirements.txt`.
    2. You may also need to check the compatibility between the `setuptools`, `Cython`, and `PyTorch` in your environment.
