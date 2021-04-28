# DETR

## Introduction

<!-- [ALGORITHM] -->

We provide the config files for Deformable DETR: [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159).

```
@inproceedings{
zhu2021deformable,
title={Deformable {\{}DETR{\}}: Deformable Transformers for End-to-End Object Detection},
author={Xizhou Zhu and Weijie Su and Lewei Lu and Bin Li and Xiaogang Wang and Jifeng Dai},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=gZ9hCDWe6ke}
}
```

## Results and Models

| Backbone | Model | Lr schd  | box AP | Config | Download |
|:------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50 | Deformable DETR  |50e  | 44.5 | [config]() | [model]() &#124; [log]() |
| R-50 | + iterative bounding box refinement  |50e  | 46.1 | [config]() | [model]() &#124; [log]() |
| R-50 | ++ two-stage Deformable DETR  |50e  | 46.8 | [config]() | [model]() &#124; [log]() |

# NOTE

1. All models are trained with batch size 32.
2. The performance is unstable. `Deformable DETR` and `iterative bounding box refinement` may fluctuate about 0.3 mAP. `two-stage Deformable DETR` fluctuate about 0.2 mAP.
