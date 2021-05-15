# YOLOv3

## Introduction

<!-- [ALGORITHM] -->

```latex
@misc{redmon2018yolov3,
    title={YOLOv3: An Incremental Improvement},
    author={Joseph Redmon and Ali Farhadi},
    year={2018},
    eprint={1804.02767},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Results and Models

|    Backbone     |  Scale  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download  |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|   DarkNet-53    |   320   |   273e  |   2.7    |      63.9      |  27.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_d53_320_273e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-20200819_172101.log.json) |
|   DarkNet-53    |   416   |   273e  |   3.8    |      61.2      |  30.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_d53_mstrain-416_273e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-20200819_173424.log.json) |
|   DarkNet-53    |   608   |   273e  |   7.1    |      48.1      |  33.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco-139f5633.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco-20200819_170820.log.json) |

## Credit

This implementation originates from the project of Haoyu Wu(@wuhy08) at Western Digital.
