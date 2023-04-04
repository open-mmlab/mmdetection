# CSPNeXt ImageNet Pre-training

In this folder, we provide the imagenet pre-training config of RTMDet's backbone CSPNeXt.

## Requirements

To train with these configs, please install [MMClassification 1.x](https://github.com/open-mmlab/mmclassification/tree/1.x) first.

Install by MIM:

```shell
mim install mmcls>=1.0.0rc0
```

or install by pip:

```shell
pip install mmcls>=1.0.0rc0
```

## Prepare Dataset

To pre-train on ImageNet, you need to prepare the dataset first. Please refer to the [guide](https://mmclassification.readthedocs.io/en/1.x/user_guides/dataset_prepare.html#imagenet).

## How to Train

You can use the classification config in the same way as the detection config.

For single-GPU training, run:

```shell
python tools/train.py \
    ${CONFIG_FILE} \
    [optional arguments]
```

For multi-GPU training, run:

```shell
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```

More details can be found in [user guides](https://mmdetection.readthedocs.io/en/latest/user_guides/train.html).

## Results and Models

|    Model     | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                                              Download                                                               |
| :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :---------------------------------------------------------------------------------------------------------------------------------: |
| CSPNeXt-tiny |  224x224   |   2.73    |   0.34   |   69.44   |   89.45   |    [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e-3a2dd350.pth)     |
|  CSPNeXt-s   |  224x224   |   4.89    |   0.66   |   74.41   |   92.23   |      [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e-ea671761.pth)      |
|  CSPNeXt-m   |  224x224   |   13.05   |   1.93   |   79.27   |   94.79   | [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth) |
|  CSPNeXt-l   |  224x224   |   27.16   |   4.19   |   81.30   |   95.62   | [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth) |
|  CSPNeXt-x   |  224x224   |   48.85   |   7.76   |   82.10   |   95.69   | [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-x_8xb256-rsb-a1-600e_in1k-b3f78edd.pth) |
