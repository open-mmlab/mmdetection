# Timm Example

> [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)

<!-- [OTHERS] -->

## Abstract

Py**T**orch **Im**age **M**odels (`timm`) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

<!--
<div align=center>
<img src="" height="400" />
</div>
-->

## Results and Models

### RetinaNet

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                          Config                           | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-------------------------------------------------------: | :------: |
|      R-50       | pytorch |   1x    |          |                |        |   [config](./retinanet_timm-tv-resnet50_fpn_1x_coco.py)   |          |
| EfficientNet-B1 |    -    |   1x    |          |                |        | [config](./retinanet_timm-efficientnet-b1_fpn_1x_coco.py) |          |

## Usage

### Install additional requirements

MMDetection supports timm backbones via `TIMMBackbone`, a wrapper class in MMClassification.
Thus, you need to install `mmcls` in addition to timm.
If you have already installed requirements for mmdet, run

```shell
pip install 'dataclasses; python_version<"3.7"'
pip install timm
pip install 'mmcls>=0.20.0'
```

See [this document](https://mmclassification.readthedocs.io/en/latest/install.html) for the details of MMClassification installation.

### Edit config

- See example configs for basic usage.
- See the documents of [timm feature extraction](https://rwightman.github.io/pytorch-image-models/feature_extraction/#multi-scale-feature-maps-feature-pyramid) and [TIMMBackbone](https://mmclassification.readthedocs.io/en/latest/api.html#mmcls.models.backbones.TIMMBackbone) for details.
- Which feature map is output depends on the backbone.
  Please check `backbone out_channels` and `backbone out_strides` in your log, and modify `model.neck.in_channels` and `model.backbone.out_indices` if necessary.
- If you use Vision Transformer models that do not support `features_only=True`, add `custom_hooks = []` to your config to disable `NumClassCheckHook`.

## Citation

```latex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```
