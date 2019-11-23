# Training-Time-Friendly Network for Real-Time Object Detection

## Introduction
We provide config files to reproduce the results in the AAAI 2020 paper [TTFNet](https://arxiv.org/pdf/1909.00700.pdf).
```
@article{liu2019training,
  title   = {Training-Time-Friendly Network for Real-Time Object Detection},
  author  = {Zili Liu, Tu Zheng, Guodong Xu, Zheng Yang, Haifeng Liu, Deng Cai},
  journal = {arXiv preprint arXiv:1909.00700},
  year    = {2019}
}
```

## Results and Models

| Backbone  | Style   | Aug | Pre train | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-18  | pytorch | N       | Y      | 1x      | 4.3 | -               | 112      | 25.9 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ttfnet/ttfnet18_1x-fe6884.pth) |
|   R-18   | pytorch |  N   |     Y     |   2x    |    -     |          -          |      112       |  28.1  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ttfnet/ttfnet18_2x-37373a.pth) |
|   R-18   | pytorch |  Y   |     Y     |   10x   |    -     |          -          |      113       |  31.7  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ttfnet/ttf18_aug_10x-0c5709be.pth) |
|   R-34   | pytorch |  N   |     Y     |   2x    |   5.1    |          -          |       86       |  31.2  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ttfnet/ttfnet34_2x-0577d0.pth) |
|   R-34   | pytorch |  Y   |     Y     |   10x   |    -     |          -          |       88       |  35.3  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ttfnet/ttf34_aug_10x-b394ba77.pth) |
| D-53 | - | N | Y | 1x | 7.5 | - | 55 | 32.9 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ttfnet/ttfnet53_1x-4811e4.pth) |
| D-53 | - | N | Y | 2x | - | - | 54 | 35.1 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ttfnet/ttfnet53_2x-b381dd.pth) |
| D-53 | - | Y | Y | 10x | - | - | 57 | 39.3 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ttfnet/ttf53_aug_10x-86c43dd3.pth) |


| Backbone  | Style   | Aug | Pre train | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-18    | pytorch | Y       | N    | 10x   | -        | -                   | 121           | 30.3 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ttfnet/ttf18_scratch_aug_10x-4dd327cf.pth) |
| R-34   | pytorch | Y       | N       | 10x    | -        | -                   | 89            | 33.2 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ttfnet/ttf34_scratch_aug_10x-da045e42.pth) |
| D-53 | -   | Y       | N       | 10x   | -     | -              | 57          | 36.2 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ttfnet/ttf53_scratch_aug_10x-56878a40.pth) |

**Notes:**

- Aug means the model is trained with the data augmentation.
- Inference time is measured on 1 GTX 1080Ti.
