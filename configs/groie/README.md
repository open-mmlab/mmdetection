# GRoIE

## A novel Region of Interest Extraction Layer for Instance Segmentation

By Leonardo Rossi, Akbar Karimi and Andrea Prati from
[IMPLab](http://implab.ce.unipr.it/).

We provide configs to reproduce the results in the paper for
"*A novel Region of Interest Extraction Layer for Instance Segmentation*"
on COCO object detection.

## Introduction

<!-- [ALGORITHM] -->

This paper is motivated by the need to overcome to the limitations of existing
RoI extractors which select only one (the best) layer from FPN.

Our intuition is that all the layers of FPN retain useful information.

Therefore, the proposed layer (called Generic RoI Extractor - **GRoIE**)
introduces non-local building blocks and attention mechanisms to boost the
performance.

## Results and models

The results on COCO 2017 minival (5k images) are shown in the below table.
You can find
[here](https://drive.google.com/drive/folders/19ssstbq_h0Z1cgxHmJYFO8s1arf3QJbT)
the trained models.

### Application of GRoIE to different architectures

| Backbone  | Method            | Lr schd | box AP | mask AP |  Config | Download|
| :-------: | :--------------: | :-----: | :----: | :-----: | :-------:| :--------:|
| R-50-FPN  | Faster Original  |   1x    |  37.4  |         | [config](../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json) |
| R-50-FPN  | + GRoIE          |   1x    |  38.3  |         | [config](./faster_rcnn_r50_fpn_groie_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/groie/faster_rcnn_r50_fpn_groie_1x_coco/faster_rcnn_r50_fpn_groie_1x_coco_20200604_211715-66ee9516.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/groie/faster_rcnn_r50_fpn_groie_1x_coco/faster_rcnn_r50_fpn_groie_1x_coco_20200604_211715.log.json) |
| R-50-FPN  | Grid R-CNN       |   1x    |  39.1  |         | [config](./grid_rcnn_r50_fpn_gn-head_1x_coco.py)| [model](http://download.openmmlab.com/mmdetection/v2.0/groie/grid_rcnn_r50_fpn_gn-head_1x_coco/grid_rcnn_r50_fpn_gn-head_1x_coco_20200605_202059-64f00ee8.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/groie/grid_rcnn_r50_fpn_gn-head_1x_coco/grid_rcnn_r50_fpn_gn-head_1x_coco_20200605_202059.log.json) |
| R-50-FPN  | + GRoIE          |   1x    |    |         | [config](./grid_rcnn_r50_fpn_gn-head_groie_1x_coco.py)||
| R-50-FPN  | Mask R-CNN       |   1x    |  38.2  |  34.7   | [config](../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py)| [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205_050542.log.json) |
| R-50-FPN  | + GRoIE          |   1x    |  39.0  |  36.0   | [config](./mask_rcnn_r50_fpn_groie_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r50_fpn_groie_1x_coco/mask_rcnn_r50_fpn_groie_1x_coco_20200604_211715-50d90c74.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r50_fpn_groie_1x_coco/mask_rcnn_r50_fpn_groie_1x_coco_20200604_211715.log.json) |
| R-50-FPN  | GC-Net           |   1x    |  40.7  |  36.5   | [config](../gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200202-50b90e5c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200202_085547.log.json) |
| R-50-FPN  | + GRoIE          |   1x    |  41.0  |  37.8   | [config](./mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco.py) |[model](http://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200604_211715-42eb79e1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200604_211715-42eb79e1.pth) |
| R-101-FPN | GC-Net           |   1x    |  42.2  |  37.8   | [config](../gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200206-8407a3f0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200206_142508.log.json) |
| R-101-FPN | + GRoIE          |   1x    |   |    | [config](./mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco.py)| [model](http://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200607_224507-8daae01c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200607_224507.log.json) |

## Citation

If you use this work or benchmark in your research, please cite this project.

```latex
@misc{rossi2020novel,
    title={A novel Region of Interest Extraction Layer for Instance Segmentation},
    author={Leonardo Rossi and Akbar Karimi and Andrea Prati},
    year={2020},
    eprint={2004.13665},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Contact

The implementation of GROI is currently maintained by
[Leonardo Rossi](https://github.com/hachreak/).
