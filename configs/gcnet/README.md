# GCNet for Object Detection

By [Yue Cao](http://yue-cao.me), [Jiarui Xu](http://jerryxu.net), [Stephen Lin](https://scholar.google.com/citations?user=c3PYmxUAAAAJ&hl=en), Fangyun Wei, [Han Hu](https://sites.google.com/site/hanhushomepage/).

This repo is a official implementation of ["GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond"](https://arxiv.org/abs/1904.11492) on COCO object detection based on open-mmlab's [mmdetection](https://github.com/open-mmlab/mmdetection).
Many thanks to mmdetection for their simple and clean framework.

## Introduction

**GCNet** is initially described in [arxiv](https://arxiv.org/abs/1904.11492). Via absorbing advantages of Non-Local Networks (NLNet) and Squeeze-Excitation Networks (SENet),  GCNet provides a simple, fast and effective approach for global context modeling, which generally outperforms both NLNet and SENet on major benchmarks for various recognition tasks.

## Citing GCNet

```
@article{cao2019GCNet,
  title={GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond},
  author={Cao, Yue and Xu, Jiarui and Lin, Stephen and Wei, Fangyun and Hu, Han},
  journal={arXiv preprint arXiv:1904.11492},
  year={2019}
}
```

## Main Results

| Backbone  |      Model       |    Context     | Lr schd | box AP | mask AP | Download  |
| :-------: | :--------------: | :------------: | :-----: | :----: | :-----: | :-------: |
| R-50-FPN  |       Mask       |       -        |   1x    |  37.2  |  33.8   | [model](https://1drv.ms/f/s!AkEXj14LxwVpcEkPyk8YWZVm9fo) |
|           |                  | GC(c3-c5, r16) |   1x    |  39.4  |  35.7   | [model](https://1drv.ms/f/s!AkEXj14LxwVpbSe8MYcjXToB090) |
|           |                  | GC(c3-c5, r4)  |   1x    |  39.9  |  36.2   | [model](https://1drv.ms/f/s!AkEXj14LxwVpaiEOLB67my2og3c) |
| R-101-FPN |       Mask       |       -        |   1x    |  39.8  |  36.0   | [model](https://1drv.ms/f/s!AkEXj14LxwVpcTx0JnBi1tuK4KU) |
|           |                  | GC(c3-c5, r16) |   1x    |  41.1  |  37.4   | [model](https://1drv.ms/f/s!AkEXj14LxwVpbqlp5GGBiZsoVaU) |
|           |                  | GC(c3-c5, r4)  |   1x    |  41.7  |  37.6   | [model](https://1drv.ms/f/s!AkEXj14LxwVpa-WTFi9kcyuWB8I) |
| X-101-FPN |       Mask       |       -        |   1x    |  41.2  |  37.3   | [model](https://1drv.ms/f/s!AkEXj14LxwVpcgwQlf8ok6IZahc) |
|           |                  | GC(c3-c5, r16) |   1x    |  42.4  |  38.0   | [model](https://1drv.ms/f/s!AkEXj14LxwVpb_4H8Q_bsmHZP7c) |
|           |                  | GC(c3-c5, r4)  |   1x    |  42.9  |  38.5   | [model](https://1drv.ms/f/s!AkEXj14LxwVpbLZR1Wu2JjMNyW0) |
| X-101-FPN |   Cascade Mask   |       -        |   1x    |  44.7  |  38.3   | [model](https://1drv.ms/f/s!Ak5O6Kz9hYOghnGJIdfM2fWlBsMR) |
|           |                  | GC(c3-c5, r16) |   1x    |  45.9  |  39.3   | [model](https://1drv.ms/f/s!Ak5O6Kz9hYOghm7gwd5yd4ZX-Dub) |
|           |                  | GC(c3-c5, r4)  |   1x    |  46.5  |  39.7   | [model](https://1drv.ms/f/s!Ak5O6Kz9hYOghk9roKJSNNpJGGsE) |
| X-101-FPN | DCN Cascade Mask |       -        |   1x    |  47.1  |  40.4   | [model](https://1drv.ms/f/s!Ak5O6Kz9hYOghmw3k3L6SbKl84Ex) |
|           |                  | GC(c3-c5, r16) |   1x    |  47.9  |  40.9   | [model](https://1drv.ms/f/s!Ak5O6Kz9hYOghm98OS64BAt0ZuOo) |
|           |                  | GC(c3-c5, r4)  |   1x    |  47.9  |  40.8   | [model](https://1drv.ms/f/s!Ak5O6Kz9hYOghlHmLvwSeF43ZISq) |

**Notes:**

- `GC` denotes Global Context (GC) block is inserted after 1x1 conv of backbone. 
- `DCN` denotes replace 3x3 conv with 3x3 Deformable Convolution in `c3-c5` stages of backbone.
- `r4` and `r16` denote ratio 4 and ratio 16 in GC block respectively. 

## Requirements

- Linux(tested on Ubuntu 16.04)
- Python 3.6+
- PyTorch 1.0.0
- Cython
- [apex](https://github.com/NVIDIA/apex) (Sync BN)

## Install

a. Install PyTorch 1.0 and torchvision following the [official instructions](https://pytorch.org/).

b. Install latest apex with CUDA and C++ extensions following this [instructions](https://github.com/NVIDIA/apex#quick-start). 
The [Sync BN](https://nvidia.github.io/apex/parallel.html#apex.parallel.SyncBatchNorm) implemented by apex is required.

c. Clone the GCNet repository. 

```bash
 git clone https://github.com/xvjiarui/GCNet.git 
```

d. Compile cuda extensions.

```bash
cd GCNet
pip install cython  # or "conda install cython" if you prefer conda
./compile.sh  # or "PYTHON=python3 ./compile.sh" if you use system python3 without virtual environments
```

e. Install GCNet version mmdetection (other dependencies will be installed automatically).

```bash
python(3) setup.py install  # add --user if you want to install it locally
# or "pip install ."
```

Note: You need to run the last step each time you pull updates from github. 
Or you can run `python(3) setup.py develop` or `pip install -e .` to install mmdetection if you want to make modifications to it frequently.

Please refer to mmdetection install [instruction](https://github.com/open-mmlab/mmdetection/blob/master/INSTALL.md) for more details.

## Environment

### Hardware

- 8 NVIDIA Tesla V100 GPUs
- Intel Xeon 4114 CPU @ 2.20GHz

### Software environment

- Python 3.6.7
- PyTorch 1.0.0
- CUDA 9.0
- CUDNN 7.0
- NCCL 2.3.5

## Usage

### Train

As in original mmdetection, distributed training is recommended for either single machine or multiple machines.

```bash
./tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> [optional arguments]
```

Supported arguments are:

- --validate: perform evaluation every k (default=1) epochs during the training.
- --work_dir <WORK_DIR>: if specified, the path in config file will be replaced.

### Evaluation

To evaluate trained models, output file is required.

```bash
python tools/test.py <CONFIG_FILE> <MODEL_PATH> [optional arguments]
```

Supported arguments are:

- --gpus: number of GPU used for evaluation
- --out: output file name, usually ends wiht `.pkl`
- --eval: type of evaluation need, for mask-rcnn, `bbox segm` would evaluate both bounding box and mask AP. 
