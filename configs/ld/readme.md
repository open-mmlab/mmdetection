# Localization Distillation for Object Detection

## Introduction

LD is the extension of knowledge distillation on localization task, which utilizes the learned bbox distributions to transfer the localization dark knowledge from teacher to student.

- [Localization Distillation for Object Detection](https://arxiv.org/abs/2102.12252)

```latex
@Article{zheng2021LD,
  title={Localization Distillation for Object Detection},
  author= {Zhaohui Zheng, Rongguang Ye, Ping Wang, Jun Wang, Dongwei Ren, Wangmeng Zuo},
  journal={arXiv:2102.12252},
  year={2021}
}
```

### GFocalV1 with LD

|  Teacher  | Student | Training schedule | Mini-batch size | AP (val) | AP50 (val) | AP75 (val) | AP (test-dev) | AP50 (test-dev) | AP75 (test-dev) | AR100 (test-dev) |
| :-------: | :-----: | :---------------: | :-------------: | :------: | :--------: | :--------: | :-----------: | :-------------: | :-------------: | :--------------: |
|    --     |  R-18   |        1x         |        6        |   35.8   |    53.1    |    38.2    |     36.0      |      53.4       |      38.7       |       55.3       |
|   R-101   |  R-18   |        1x         |        6        |   36.5   |    52.9    |    39.3    |     36.8      |      53.5       |      39.9       |       56.6       |
|    --     |  R-34   |        1x         |        6        |   38.9   |    56.6    |    42.2    |     39.2      |      56.9       |      42.3       |       58.0       |
|   R-101   |  R-34   |        1x         |        6        |   39.8   |    56.6    |    43.1    |     40.0      |      57.1       |      43.5       |       59.3       |
|    --     |  R-50   |        1x         |        6        |   40.1   |    58.2    |    43.1    |     40.5      |      58.8       |      43.9       |       59.0       |
|   R-101   |  R-50   |        1x         |        6        |   41.1   |    58.7    |    44.9    |     41.2      |      58.8       |      44.7       |       59.8       |
|    --     |  R-101  |        2x         |        6        |   44.6   |    62.9    |    48.4    |     45.0      |      63.6       |      48.9       |       62.3       |
| R-101-DCN |  R-101  |        2x         |        6        |   45.4   |    63.1    |    49.5    |     45.6      |      63.7       |      49.8       |       63.3       |
