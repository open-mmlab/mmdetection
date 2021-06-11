# DetectoRS

## Introduction

<!-- [ALGORITHM] -->

We provide the config files for [DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution](https://arxiv.org/pdf/2006.02334.pdf).

```BibTeX
@article{qiao2020detectors,
  title={DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution},
  author={Qiao, Siyuan and Chen, Liang-Chieh and Yuille, Alan},
  journal={arXiv preprint arXiv:2006.02334},
  year={2020}
}
```

## Dataset

DetectoRS requires COCO and [COCO-stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) dataset for training. You need to download and extract it in the COCO dataset path.
The directory should be like this.

```none
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
|   |   ├── stuffthingmaps
```

## Results and Models

DetectoRS includes two major components:

- Recursive Feature Pyramid (RFP).
- Switchable Atrous Convolution (SAC).

They can be used independently.
Combining them together results in DetectoRS.
The results on COCO 2017 val are shown in the below table.

| Method | Detector | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:------:|:--------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| RFP | Cascade + ResNet-50 | 1x | 7.5 | - | 44.8 | | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors/cascade_rcnn_r50_rfp_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_rfp_1x_coco/cascade_rcnn_r50_rfp_1x_coco-8cf51bfd.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_rfp_1x_coco/cascade_rcnn_r50_rfp_1x_coco_20200624_104126.log.json) |
| SAC | Cascade + ResNet-50 | 1x | 5.6 | - | 45.0| | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors/cascade_rcnn_r50_sac_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_sac_1x_coco/cascade_rcnn_r50_sac_1x_coco-24bfda62.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_sac_1x_coco/cascade_rcnn_r50_sac_1x_coco_20200624_104402.log.json) |
| DetectoRS | Cascade + ResNet-50 | 1x | 9.9 | - | 47.4 | | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_cascade_rcnn_r50_1x_coco/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_cascade_rcnn_r50_1x_coco/detectors_cascade_rcnn_r50_1x_coco_20200706_001203.log.json) |
| RFP | HTC + ResNet-50 | 1x | 11.2 | - | 46.6 | 40.9 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors/htc_r50_rfp_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_rfp_1x_coco/htc_r50_rfp_1x_coco-8ff87c51.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_rfp_1x_coco/htc_r50_rfp_1x_coco_20200624_103053.log.json) |
| SAC | HTC + ResNet-50 | 1x | 9.3 | - | 46.4 | 40.9 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors/htc_r50_sac_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_sac_1x_coco/htc_r50_sac_1x_coco-bfa60c54.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_sac_1x_coco/htc_r50_sac_1x_coco_20200624_103111.log.json) |
| DetectoRS | HTC + ResNet-50 | 1x | 13.6 | - | 49.1 | 42.6 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors/detectors_htc_r50_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco-329b1453.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco_20200624_103659.log.json) |
| DetectoRS | HTC + ResNet-101 | 20e | 19.6 |  | 50.5 | 43.9 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors/detectors_htc_r101_20e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r101_20e_coco/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r101_20e_coco/detectors_htc_r101_20e_coco_20210419_203638.log.json) |

*Note*: This is a re-implementation based on MMDetection-V2.
The original implementation is based on MMDetection-V1.
