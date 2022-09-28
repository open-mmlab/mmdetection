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

More details can be found in [user guides](https://mmdetection.readthedocs.io/en/3.x/user_guides/train.html).

## Results and Models

|    Model     | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                                      Download                                                       |
| :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :-----------------------------------------------------------------------------------------------------------------: |
| CSPNeXt-tiny |  224x224   |   2.73    |  0.339   |   69.44   |   89.45   | [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth) |
|  CSPNeXt-s   |  224x224   |   4.89    |  0.664   |   74.41   |   92.23   |  [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth)   |
