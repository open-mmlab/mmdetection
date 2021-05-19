# CentripetalNet

## Introduction

<!-- [ALGORITHM] -->

```latex
@InProceedings{Dong_2020_CVPR,
author = {Dong, Zhiwei and Li, Guoxuan and Liao, Yue and Wang, Fei and Ren, Pengju and Qian, Chen},
title = {CentripetalNet: Pursuing High-Quality Keypoint Pairs for Object Detection},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Results and models

| Backbone        | Batch Size | Step/Total Epochs | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :--------: |:----------------: | :------: | :------------: | :----: | :------: | :--------: |
| HourglassNet-104 | [16 x 6](./centripetalnet_hourglass104_mstest_16x6_210e_coco.py) | 190/210 | 16.7 | 3.7 | 44.8 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco/centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804-3ccc61e5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco/centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804.log.json) |

Note:

- TTA setting is single-scale and `flip=True`.
- The model we released is the best checkpoint rather than the latest checkpoint (box AP 44.8 vs 44.6 in our experiment).
