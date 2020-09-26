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
| RFP | Cascade + ResNet-50 | 1x | 7.5 | - | 44.8 | | [model](http://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_rfp_1x_coco/cascade_rcnn_r50_rfp_1x_coco-8cf51bfd.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_rfp_1x_coco/cascade_rcnn_r50_rfp_1x_coco_20200624_104126.log.json) |
| SAC | Cascade + ResNet-50 | 1x | 5.6 | - | 45.0| | [model](http://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_sac_1x_coco/cascade_rcnn_r50_sac_1x_coco-24bfda62.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_sac_1x_coco/cascade_rcnn_r50_sac_1x_coco_20200624_104402.log.json) |
| DetectoRS | Cascade + ResNet-50 | 1x | 9.9 | - | 47.4 | | [model](http://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_cascade_rcnn_r50_1x_coco/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_cascade_rcnn_r50_1x_coco/detectors_cascade_rcnn_r50_1x_coco_20200706_001203.log.json) |
| RFP | HTC + ResNet-50 | 1x | 11.2 | - | 46.6 | 40.9 | [model](http://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_rfp_1x_coco/htc_r50_rfp_1x_coco-8ff87c51.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_rfp_1x_coco/htc_r50_rfp_1x_coco_20200624_103053.log.json) |
| SAC | HTC + ResNet-50 | 1x | 9.3 | - | 46.4 | 40.9 | [model](http://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_sac_1x_coco/htc_r50_sac_1x_coco-bfa60c54.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_sac_1x_coco/htc_r50_sac_1x_coco_20200624_103111.log.json) |
| DetectoRS | HTC + ResNet-50 | 1x | 13.6 | - | 49.1 | 42.6 | [model](http://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco-329b1453.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco_20200624_103659.log.json) |

*Note*: This is a re-implementation based on MMDetection-V2.
The original implementation is based on MMDetection-V1.
