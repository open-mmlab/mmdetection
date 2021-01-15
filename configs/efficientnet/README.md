# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

## Introduction

We implement EfficientNet models in detection systems and provide their first results on Mask R-CNN, Faster R-CNN and RetinaNet.

The pre-trained modles are converted from [model zoo of pycls](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md).

```
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

For more general usage, we also provide script `efficientnet2mmdet.py` in the tools directory to convert the key of models pretrained by [pycls](https://github.com/facebookresearch/pycls/) to
EfficientNet-style checkpoints used in MMDetection.

```bash
python -u tools/efficientnet2mmedet.py ${PRETRAIN_PATH} ${STORE_PATH}
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

## Results
