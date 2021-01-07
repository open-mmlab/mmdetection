# SWA Object Detection

## Introduction

[Stochastic Weights Averaging](https://arxiv.org/abs/1803.05407) (SWA) can help improve generalization of deep networks and is found also very effective in object detection. Through extensive experiments, we discover a good policy of applying it to object detection, which can consistently achieve ~1.0 AP improvement over various object detectors on the COCO benchmark.  The recipe is surprisingly simple: **training your detector for an extra 12 epochs using cyclical learning rates and then averaging these 12 checkpoints as your final detection model**, which does not incur any inference cost and is really worth trying. The full report is available at: [https://arxiv.org/abs/2012.12645](https://arxiv.org/abs/2012.12645).

## Citing SWA Object Detection

```latex
@article{zhang2020swa,
  title={SWA Object Detection},
  author={Zhang, Haoyang and Wang, Ying and Dayoub, Feras and S{\"u}nderhauf, Niko},
  journal={arXiv preprint arXiv:2012.12645},
  year={2020}
}
```

## Results and Models

| Model                                             | bbox AP (val)     | segm AP  (val)      | &nbsp; &nbsp; Download  &nbsp; &nbsp;  |
|:-------------------------------------------------:|:-----------------:|:-------------------:|:--------------------------------------:|
| SWA-MaskRCNN-R50-1x-0.02-0.0002-38.2-34.7         | 39.1, +0.9        | 35.5, +0.8          | [model](https://drive.google.com/file/d/1Pue5dYW_JkQXwRQmBSeSX6cVluVBQYFB/view?usp=sharing) &#124; [config](configs/swa/swa_mask_rcnn_r50_fpn_1x_coco.py) |
| SWA-MaskRCNN-R101-1x-0.02-0.0002-40.0-36.1        | 41.0, +1.0        | 37.0, +0.9          | [model](https://drive.google.com/file/d/1_qAfKqs4aNND_IjWUMyirV7e9MVpFiHy/view?usp=sharing) &#124; [config](configs/swa/swa_mask_rcnn_r101_fpn_1x_coco.py) |
| SWA-MaskRCNN-R101-2x-0.02-0.0002-40.8-36.6        | 41.7, +0.9        | 37.4, +0.8          | [model](https://drive.google.com/file/d/1i1NjfU9F-GdUBc-nRcJ60Ap1BfS-avyH/view?usp=sharing) &#124; [config](configs/swa/swa_mask_rcnn_r101_fpn_2x_coco.py) |
| SWA-FasterRCNN-R50-1x-0.02-0.0002-37.4            | 38.4, +1.0        | -                   | [model](https://drive.google.com/file/d/1wbPDsas-nmBvctsebGa05I33-iP2kCyV/view?usp=sharing) &#124; [config](configs/swa/swa_faster_rcnn_r50_fpn_1x_coco.py) |
| SWA-FasterRCNN-R101-1x-0.02-0.0002-39.4           | 40.3, +0.9        | -                   | [model](https://drive.google.com/file/d/112CzQ5nCTz_RkDSkGdbCT3pmu3ajsvxT/view?usp=sharing) &#124; [config](configs/swa/swa_faster_rcnn_r101_fpn_1x_coco.py) |
| SWA-FasterRCNN-R101-2x-0.02-0.0002-39.8           | 40.7, +0.9        | -                   | [model](https://drive.google.com/file/d/1hTS5cVebeDaC16W3laRvAtog5a3ZPFZF/view?usp=sharing) &#124; [config](configs/swa/swa_faster_rcnn_r101_fpn_2x_coco.py) |
| SWA-RetinaNet-R50-1x-0.01-0.0001-36.5             | 37.8, +1.3        | -                   | [model](https://drive.google.com/file/d/1IYVpS6AsfXzwTM80P2ZaWS-u1FXa63Xd/view?usp=sharing) &#124; [config](configs/swa/swa_retinanet_r50_fpn_1x_coco.py) |
| SWA-RetinaNet-R101-1x-0.01-0.0001-38.5            | 39.7, +1.2        | -                   | [model](https://drive.google.com/file/d/1UMLmqwJ29DFOr52JJp_sTD5Tx5nzmFO2/view?usp=sharing) &#124; [config](configs/swa/swa_retinanet_r101_fpn_1x_coco.py) |
| SWA-RetinaNet-R101-2x-0.01-0.0001-38.9            | 40.0, +1.1        | -                   | [model](https://drive.google.com/file/d/11Z4Vnm8175WM6xy-0qu_1KPhoWAEPhH8/view?usp=sharing) &#124; [config](configs/swa/swa_retinanet_r101_fpn_2x_coco.py) |
| SWA-FCOS-R50-1x-0.01-0.0001-36.6                  | 38.0, +1.4        | -                   | [model](https://drive.google.com/file/d/1xgxXYYT3p-Azv5PnXMrPAKBtN1VhvBWz/view?usp=sharing) &#124; [config](configs/swa/swa_fcos_r50_caffe_fpn_4x4_1x_coco.py) |
| SWA-FCOS-R101-1x-0.01-0.0001-39.2                 | 40.3, +1.1        | -                   | [model](https://drive.google.com/file/d/1GYx79UpPaL321Va0jSZ4DDWkms00pMs0/view?usp=sharing) &#124; [config](configs/swa/swa_fcos_r101_caffe_fpn_gn-head_4x4_1x_coco.py) |
| SWA-FCOS-R101-2x-0.01-0.0001-39.1                 | 40.2, +1.1        | -                   | [model](https://drive.google.com/file/d/1yiyeXSkuFMq0FVvrjJkcN72etd1p3gKp/view?usp=sharing) &#124; [config](configs/swa/swa_fcos_r101_caffe_fpn_gn-head_4x4_2x_coco.py) |
| SWA-YOLOv3(320)-D53-273e-0.001-0.00001-27.9       | 28.7, +0.8        | -                   | [model](https://drive.google.com/file/d/1pZS52yHoQVh6kybV9D83bvzsSIN_wFvh/view?usp=sharing) &#124; [config](configs/swa/swa_yolov3_d53_320_273e_coco.py) |
| SWA-YOLOv3(680)-D53-273e-0.001-0.00001-33.4       | 34.2, +0.8        | -                   | [model](https://drive.google.com/file/d/1FQd0oWsk18VKUv40ywtYe4G0SiYpqKZ7/view?usp=sharing) &#124; [config](configs/swa/swa_yolov3_d53_mstrain-608_273e_coco.py) |
| SWA-VFNet-R50-1x-0.01-0.0001-41.6                 | 42.8, +1.2        | -                   | [model](https://drive.google.com/file/d/15nRdpQqjiCzuvLYytdJg86g9x4tQ8cCo/view?usp=sharing) &#124; [config](configs/swa/swa_vfnet_r50_fpn_1x_coco.py) |
| SWA-VFNet-R101-1x-0.01-0.0001-43.0                | 44.3, +1.3        | -                   | [model](https://drive.google.com/file/d/13tGeJBqzS1ITAaESMMDAaEHMrSJvA0g2/view?usp=sharing) &#124; [config](configs/swa/swa_vfnet_r101_fpn_1x_coco.py) |
| SWA-VFNet-R101-2x-0.01-0.0001-43.5                | 44.5, +1.0        | -                   | [model](https://drive.google.com/file/d/1rJm0tcxWIjCDl6Np7MPR3OE6mwUDWwh0/view?usp=sharing) &#124; [config](configs/swa/swa_vfnet_r101_fpn_2x_coco.py) |

**Notes:**

- SWA-MaskRCNN-R50-1x-0.02-0.0002-38.2-34.7 means this SWA model is produced based on the pre-trained Mask RCNN model that has a ResNet50 backbone, is trained under 1x schedule with the initial learning rate 0.02 and ending learning rate 0.0002, and achieves 38.2 bbox AP and 34.7 mask AP on the COCO val2017 respectively. This SWA model acheives 39.1 bbox AP and 35.5 mask AP, which are higher than the pre-trained model by 0.9 bbox AP and 0.8 mask AP respectively. This rule applies to other object detectors.
- More detailed results can be found at [swa_object_detection-Github](https://github.com/hyz-xmaster/swa_object_detection).
