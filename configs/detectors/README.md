# DetectoRS

## Introduction

We provide the config files for [DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution](https://arxiv.org/pdf/2006.02334.pdf).

```BibTeX
@article{qiao2020detectors,
  title={DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution},
  author={Qiao, Siyuan and Chen, Liang-Chieh and Yuille, Alan},
  journal={arXiv preprint arXiv:2006.02334},
  year={2020}
}
```

## Results and Models

DetectoRS includes two major components:

- Recursive Feature Pyramid (RFP).
- Switchable Atrous Convolution (SAC).

They can be used independently.
Combining them together results in DetectoRS.
The results on COCO 2017 val are shown in the below table.

| Method | Detector | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
|:------:|:--------:|:-------:|:--------:|:--------------:|:------:|:-------:|:--------:|
| RFP | Cascade + ResNet-50 | 1x | | | | | |
| SAC | Cascade + ResNet-50 | 1x | | | | | |
| DetectoRS | Cascade + ResNet-50 | 1x | | | | | |
| RFP | HTC + ResNet-50 | 1x | | | | | |
| SAC | HTC + ResNet-50 | 1x | | | | | |
| DetectoRS | HTC + ResNet-50 | 1x | | | | | | |

*Note*: This is a re-implementation based on MMDetection-V2.
The original implementation is based on MMDetection-V1.
