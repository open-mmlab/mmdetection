# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

## Introduction

<!-- [ALGORITHM] -->

We implement EfficientNet models in detection systems.

The pre-trained modles are converted from [model zoo of pycls](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md).

```latex
@article{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc V},
  journal={arXiv preprint arXiv:1905.11946},
  year={2019}
}
```

## Usage

To use a efficientnet model, there are two steps to do:

1. Convert the model to EfficientNet-style supported by MMDetection

2. Modify backbone and neck in config accordingly

### Convert model

We already prepare models of FLOPs from B0 to B5 in our model zoo.

For more general usage, we also provide script `efficientnet2mmdet.py` in the tools/model_converters directory to convert the key of models pretrained by [pycls](https://github.com/facebookresearch/pycls/) to
EfficientNet-style checkpoints used in MMDetection.

```bash
python -u tools/model_converters/efficientnet2mmedet.py ${PRETRAIN_PATH} ${STORE_PATH}
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

## Results

### Results on EfficientNet-B0/B3

| Backbone  | Style   |  Lr schd | Epoch | box AP | config | model |
|:---------:|:-------:|:-------:|:--------------:|:------:|:------:|:------:|
| efficientNet-b0 | pytorch | 0.800      | 300      | 33.6    | -     | - |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219_223025.log.json) |
| efficientNet-b3| pytorch | 0.875       | 300       | 40.0     | -      | - |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_r50_fpn_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_2x_coco/fovea_r50_fpn_4x4_2x_coco_20200203-2df792b1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_2x_coco/fovea_r50_fpn_4x4_2x_coco_20200203_112043.log.json) |
