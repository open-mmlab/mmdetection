# Localization Distillation for Object Detection

## Introduction

[ALGORITHM]

```latex
@Article{zheng2021LD,
  title={Localization Distillation for Object Detection},
  author= {Zhaohui Zheng, Rongguang Ye, Ping Wang, Jun Wang, Dongwei Ren, Wangmeng Zuo},
  journal={arXiv:2102.12252},
  year={2021}
}
```

### GFocalV1 with LD

|  Teacher  | Student | Training schedule | Mini-batch size | AP (val) | AP50 (val) | AP75 (val) | Config |
| :-------: | :-----: | :---------------: | :-------------: | :------: | :--------: | :--------: |  :--------------: |
|    --     |  R-18   |        1x         |        6        |   35.8   |    53.1    |    38.2    |          |
|   R-101   |  R-18   |        1x         |        6        |   36.5   |    52.9    |    39.3    |   [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ld/ld_r18_gflv1_r101_fpn_coco_1x.py)          |
|    --     |  R-34   |        1x         |        6        |   38.9   |    56.6    |    42.2    |          |
|   R-101   |  R-34   |        1x         |        6        |   39.8   |    56.6    |    43.1    |     [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ld/ld_r34_gflv1_r101_fpn_coco_1x.py)        |
|    --     |  R-50   |        1x         |        6        |   40.1   |    58.2    |    43.1    |            |
|   R-101   |  R-50   |        1x         |        6        |   41.1   |    58.7    |    44.9    |    [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ld/ld_r50_gflv1_r101_fpn_coco_1x.py)        |
|    --     |  R-101  |        2x         |        6        |   44.6   |    62.9    |    48.4    |           |
| R-101-DCN |  R-101  |        2x         |        6        |   45.4   |    63.1    |    49.5    | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ld/ld_r101_gflv1_r101dcn_fpn_coco_1x.py)           |

## Note

- Meaning of Config name: ld_r18(student model)_gflv1(based on gflv1)_r101(teacher model)_fpn(neck)_coco(dataset)_1x(12 epoch).py
