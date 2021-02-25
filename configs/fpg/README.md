# Feature Pyramid Grids

## Introduction

```latex
@article{chen2020feature,
  title={Feature pyramid grids},
  author={Chen, Kai and Cao, Yuhang and Loy, Chen Change and Lin, Dahua and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2004.03580},
  year={2020}
}
```

## Results and Models

We benchmark the new training schedule (crop training, large batch, unfrozen BN, 50 epochs) introduced in NAS-FPN.
All backbones are Resnet-50 in pytorch style.

| Method       | Neck        | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:------------:|:-----------:|:-------:|:--------:|:--------------:|:------:|:-------:|:-------:|:--------:|
| Faster R-CNN | FPG         | 50e     | 13.2     | -              | 42.2   | -       |[config]() |
| Faster R-CNN | FPG-cha128  | 50e     | 13.2     | -              | 41.2   | -       |[config]() |
| Mask R-CNN   | FPG         | 50e     | 23.2     | -              | 42.7   | 37.8    |[config]() |
| Mask R-CNN   | FPG-cha128  | 50e     | 15.3     | -              | 41.7   | 36.9    |[config]() |
| RetinaNet    | FPG         | 50e     | 13.2     | -              |        | -       |[config]() |
| RetinaNet    | FPG-cha128  | 50e     | 13.2     | -              |        | -       |[config]() |

**Note**: Cha128 means to decrease the number of channels of features and convs from 256 (default) to 128 in
Neck and BBox Head, which can greatly decrease memory consumption without sacrificing much precision.
