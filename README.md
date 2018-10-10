
# mmdetection

## Introduction

`mmdetection` is an open source object detection toolbox based on PyTorch. It is
a part of the open-mmlab project developed by Multimedia Laboratory, CUHK.

### Major features

- **Modular Design**

  One can easily construct a customized object detection framework by combining different components. 
  
- **Support of multiple frameworks out of box**

  The toolbox directly supports popular detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc. (see the release plan for more)

- **Efficient**

  All basic bbox and mask operations run on GPUs now.
  The training speed is about 5% ~ 20% faster than Detectron for different models.
  
- **State of the art**

  This was the codebase of the *MMDet* team, who won the [COCO Detection 2018 challenge](http://cocodataset.org/#detection-leaderboard). 

Apart from mmdetection, we also released a library `mmcv` for computer vision research,
which is heavily depended on by mmdetection.

## License

This project is released under the [GPLv3 license](LICENSE).

## Benchmark and model zoo

We provide our baseline results and the comparision with other popular detection projects
such as Detectron. Results and models are available in the [Model zoo](MODEL_ZOO.md).

## Installation

Requirements:

- Linux (tested on Ubuntu 16.04 and CentOS 7.2)
- Python 2.7+ or 3.4+
- PyTorch 0.4.1 and torchvision
- OpenCV
- [mmcv](https://github.com/open-mmlab/mmcv)

> Note: Though mmdetection is compatible with Python 2/3, python 3 is recommended and we do not promise future support for Python 2.

Clone the Detectron repository.

```shell
git clone https://github.com/open-mmlab/mmdetection.git
```

Compile cuda extensions.

```shell
cd mmdetection
./compile.sh  # or "PYTHON=python3 ./compile.sh" if you use system python3 without virtual environments
```

Install mmdetection (other dependencies will be installed automatically).

```shell
python(3) setup.py install  # add --user if you want to install it locally
# or "pip install ."
```

> Note: You need to run the installing step each time you pull updates from github. Git commit id will be written to the version number and also saved in trained models.

Prepare COCO dataset

It is recommended to symlink the dataset root to `$MMDETECTION/data`.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017

```


## Inference with pretrained models

To test a dataset and save the results.

```shell
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --gpus <GPU_NUM> --out <OUT_FILE>
```

To perform evaluation after testing, add `--eval <EVAL_TYPES>`. Supported types are:

- proposal_fast: eval recalls of proposals with our own codes. (supposed to get the same results as the official evaluation)
- proposal: eval recalls of proposals with the official code provided by COCO.
- bbox: eval box AP with the official code provided by COCO.
- segm: eval mask AP with the official code provided by COCO.
- keypoints: eval keypoint AP with the official code provided by COCO.

For example, to evaluate Mask R-CNN with 8 GPUs and save the result as results.pkl.

```shell
python tools/test.py configs/mask_rcnn_r50_fpn_1x.py <CHECKPOINT_FILE> --gpus 8 --out results.pkl --eval bbox segm
```

Note: Multiple GPU testing cannot achieves linear acceleration.

We also provide the ability to visualize the results when testing. Add the argument `--show` as below.

```shell
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --show
```


## Train a model

`mmdetection` implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

We suggest using distributed training even on a single machine, which is faster,
and non-distributed training are left for debugging or other purposes.

### Distributed training

`mmdetection` potentially supports multiple launch methods, e.g., PyTorch’s built-in launch utility, slurm and MPI.

We provide a training script using the launch utility provided by PyTorch.

```shell
./tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> [optional arguments]
```

Supported arguments are:

- --validate: perform evaluation every k (default=1) epochs during the training.
- --work_dir <WORK_DIR>: if specified, the path in config file will be overwritten.

### Non-distributed training

```shell
python tools/train.py <CONFIG_FILE> --gpus <GPU_NUM> --work_dir <WORK_DIR> --validate
```

Expected results in WORK_DIR:

- log file
- saved checkpoints (every k epochs, defaults=1)
- a symbol link to the latest checkpoint


## High-level APIs

We are working on a set of high-level APIs to make it more convenient to
integrate mmdetection into other projects or act as a hands-on tool for
beginners.
