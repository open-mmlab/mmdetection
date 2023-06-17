# CentripetalNet

> [CentripetalNet: Pursuing High-quality Keypoint Pairs for Object Detection](https://arxiv.org/abs/2003.09119)

<!-- [ALGORITHM] -->

## Abstract

Keypoint-based detectors have achieved pretty-well performance. However, incorrect keypoint matching is still widespread and greatly affects the performance of the detector. In this paper, we propose CentripetalNet which uses centripetal shift to pair corner keypoints from the same instance. CentripetalNet predicts the position and the centripetal shift of the corner points and matches corners whose shifted results are aligned. Combining position information, our approach matches corner points more accurately than the conventional embedding approaches do. Corner pooling extracts information inside the bounding boxes onto the border. To make this information more aware at the corners, we design a cross-star deformable convolution network to conduct feature adaption. Furthermore, we explore instance segmentation on anchor-free detectors by equipping our CentripetalNet with a mask prediction module. On MS-COCO test-dev, our CentripetalNet not only outperforms all existing anchor-free detectors with an AP of 48.0% but also achieves comparable performance to the state-of-the-art instance segmentation approaches with a 40.2% MaskAP.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143873955-42804e0e-3638-4c5b-8bf4-ac8133bbcdc8.png"/>
</div>

## Results and Models

|     Backbone     |                                Batch Size                                 | Step/Total Epochs | Mem (GB) | Inf time (fps) | box AP |                                  Config                                   |                                                                                                                                                                                                    Download                                                                                                                                                                                                    |
| :--------------: | :-----------------------------------------------------------------------: | :---------------: | :------: | :------------: | :----: | :-----------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| HourglassNet-104 | [16 x 6](./centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py) |      190/210      |   16.7   |      3.7       |  44.8  | [config](./centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco/centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804-3ccc61e5.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco/centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804.log.json) |

Note:

- TTA setting is single-scale and `flip=True`. If you want to reproduce the TTA performance, please add `--tta` in the test command.
- The model we released is the best checkpoint rather than the latest checkpoint (box AP 44.8 vs 44.6 in our experiment).

## Citation

```latex
@InProceedings{Dong_2020_CVPR,
author = {Dong, Zhiwei and Li, Guoxuan and Liao, Yue and Wang, Fei and Ren, Pengju and Qian, Chen},
title = {CentripetalNet: Pursuing High-Quality Keypoint Pairs for Object Detection},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
