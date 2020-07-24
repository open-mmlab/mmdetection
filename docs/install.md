## Installation

### Requirements

- Linux or macOS (Windows is not currently officially supported)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv)

### Install mmdetection

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

```python
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

```python
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

If you build PyTorch from source instead of installing the prebuilt pacakge,
you can use more CUDA versions such as 9.0.

c. Install mmcv, we recommend you to install the pre-build mmcv as below.

```shell
pip install mmcv-full==latest+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
```

See [here]((https://github.com/open-mmlab/mmcv#install-with-pip)) for different versions of MMCV compatible to different PyTorch and CUDA versions.
Optionally you can choose to compile mmcv from source by the following command

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
cd ..
```

Or directly run

```shell
pip install mmcv-full
```

**Important**:

1. The required versions of MMCV for different versions of MMDetection since V2.0 are as below. Please install the correct version of MMCV to avoid installation issues.

| MMDetection version |    MMCV version     |
|:-------------------:|:-------------------:|
| master              | mmcv-full>=1.0.2    |
| 2.3.0rc0            | mmcv-full>=1.0.2    |
| 2.2.1               | mmcv==0.6.2         |
| 2.2.0               | mmcv==0.6.2         |
| 2.1.0               | mmcv>=0.5.9, <=0.6.1|
| 2.0.0               | mmcv>=0.5.1, <=0.5.8|

2. You need to run `pip unisntall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

d. Clone the mmdetection repository.

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
```

e. Install build requirements and then install mmdetection.
(We install our forked version of pycocotools via the github repo instead of pypi
for better compatibility with our repo.)

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

If you build mmdetection on macOS, replace the last command with

```shell
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```

Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

    > Important: Be sure to remove the `./build` folder if you reinstall mmdet with a different CUDA/PyTorch version.

    ```shell
    pip uninstall mmdet
    rm -rf ./build
    find . -name "*.so" | xargs rm
    ```

2. Following the above instructions, mmdetection is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

4. Some dependencies are optional. Simply running `pip install -v -e .` will only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.

### Install with CPU only

The code can be built for CPU only environment (where CUDA isn't available).

In CPU mode you can run the demo/webcam_demo.py for example.
However some functionality is gone in this mode:

- Deformable Convolution
- Deformable ROI pooling
- CARAFE: Content-Aware ReAssembly of FEatures
- nms_cuda
- sigmoid_focal_loss_cuda

So if you try to run inference with a model containing deformable convolution you will get an error.
Note: We set `use_torchvision=True` on-the-fly in CPU mode for `RoIPool` and `RoIAlign`

### Another option: Docker Image

We provide a [Dockerfile](https://github.com/open-mmlab/mmdetection/blob/master/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.5, CUDA 10.1
docker build -t mmdetection docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data mmdetection
```

### A from-scratch setup script

Here is a full script for setting up mmdetection with conda.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y

# install the latest mmcv
pip install mmcv-full

# install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

### Using multiple MMDetection versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMDetection in the current directory.

To use the default MMDetection installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

### Common issues

We list some common issues and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them.

<details>
<summary>
Compatibility issue between MMCV and MMDetection; "ConvWS is already registered in conv layer";
</summary>
    Please install the correct version of MMCV for the version of your MMDetection following the instruction above.
</details>

<details>
<summary>
"No module named 'mmcv.ops'"; "No module named 'mmcv._ext'".
</summary>

1. Uninstall existing mmcv in the environment using `pip uninstall mmcv`.
2. Install mmcv-full following the instruction above.

</details>

<details>
<summary>
"invalid device function" or "no kernel image is available for execution".
</summary>

1. Check the CUDA compute capability of you GPU.
2. Run `python mmdet/utils/collect_env.py` to check whether PyTorch, torchvision, and MMCV are built for the correct GPU architecture. You may need to set `TORCH_CUDA_ARCH_LIST` to reinstall MMCV. The compatibility issue could happen when using old GPUS, e.g., Tesla K80 (3.7) on colab.
3. Check whether the running environment is the same as that when mmcv/mmdet is compiled. For example, you may compile mmcv using CUDA 10.0 bug run it on CUDA9.0 environments.

</details>
