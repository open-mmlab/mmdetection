# YOLOX-PAI

> [YOLOX-PAI: An Improved YOLOX, Stronger and Faster than YOLOv6](https://arxiv.org/abs/2208.13040)

<!-- [ALGORITHM] -->

## Abstract

We develop an all-in-one computer vision toolbox named EasyCV to facilitate the use of various SOTA computer vision methods. Recently, we add YOLOX-PAI, an improved version of YOLOX, into EasyCV. We conduct ablation studies to investigate the influence of some detection methods on YOLOX. We also provide an easy use for PAI-Blade which is used to accelerate the inference process based on BladeDISC and TensorRT. Finally, we receive 42.8 mAP on COCO dateset within 1.0 ms on a single NVIDIA V100 GPU, which is a bit faster than YOLOv6. A simple but efficient predictor api is also designed in EasyCV to conduct end2end object detection.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/189808824-094c66f7-f95c-4e31-8a1e-50515fce545d.png"/>
</div>

## Results and Models

|  Backbone   | ASFF | TOOD | box AP |                                                         Config                                                          |         Download         |
| :---------: | :--: | :--: | :----: | :---------------------------------------------------------------------------------------------------------------------: | :----------------------: |
| YOLOX-PAI-s |  N   |  N   |  41.8  |      [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yoloxpai/yolox_pai_s_8x8_300e_coco.py)      | [model](<>) \| [log](<>) |
| YOLOX-PAI-s |  Y   |  N   |  42.8  |   [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yoloxpai/yolox_pai_asff_s_8x8_300e_coco.py)    | [model](<>) \| [log](<>) |
| YOLOX-PAI-s |  Y   |  Y   |  43.6  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yoloxpai/yolox_pai_asff_tood_s_8x8_300e_coco.py) | [model](<>) \| [log](<>) |

## Usage

### Install additional requirements

RepVGG backbone needs to install [MMClassification](https://github.com/open-mmlab/mmclassification) first, which has abundant backbones for downstream tasks.
If you have already installed requirements for mmdet, run

```shell
pip install 'mmcls>=0.24.0'
```

See [this document](https://mmclassification.readthedocs.io/en/latest/install.html) for the details of MMClassification installation.

Minimum required version of MMCV is `1.6.3`.
See [this document](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for the details of MMCV installation.
