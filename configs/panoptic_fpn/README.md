# Panoptic feature pyramid networks
## Introduction

<!-- [ALGORITHM] -->

```
@inproceedings{kirillov2018panopticfpn,
  author = {
    Alexander Kirillov,
    Ross Girshick,
    Kaiming He,
    Piotr Dollar,
  },
  title = {},
  booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```

## Results and Models

| Backbone      | style      | Lr schd | Mem (GB) | Inf time (fps) | PQ   |  SQ  |  RQ  | PQ_th | SQ_th | RQ_th | PQ_st | SQ_st | RQ_st | Config | Download |
|:-------------:|:----------:|:-------:|:--------:|:--------------:|:----:|:----:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:------:|:--------:|
| R-50-FPN      | pytorch    | 1x      |   4.7    |                | 40.2 | 77.8 | 49.3 | 47.8  | 80.9  | 57.5  | 28.9  | 73.1  | 37.0  |        |          |
| R-50-FPN      | pytorch    | 3x      |   -      |     -          |      |      |      |       |       |       |       |       |       |        |          |
| R-101-FPN     | pytorch    | 1x      |   6.7    |                | 42.2 | 78.3 | 51.4 | 50.1  | 81.4  | 59.9  | 30.3  | 73.63 | 38.5  |        |          |
| R-101-FPN     | pytorch    | 3x      |   -      |     -          |      |      |      |       |       |       |       |       |       |        |          |
