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
|   DarkNet-53    |   320   |   273e  |   2.7    |      63.9      |  27.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_d53_320_273e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-20200819_172101.log.json) |
|   DarkNet-53    |   416   |   273e  |   3.8    |      61.2      |  30.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_d53_mstrain-416_273e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-20200819_173424.log.json) |
|   DarkNet-53    |   608   |   273e  |   7.4    |      48.1      |  33.7  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020.log.json) |

## Mixed Precision Training

We also train YOLOv3 with mixed precision training.

|    Backbone     |  Scale  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download  |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|   DarkNet-53    |   608   |   273e  |   4.7    |      48.1      |  33.8  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_d53_fp16_mstrain-608_273e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_fp16_mstrain-608_273e_coco/yolov3_d53_fp16_mstrain-608_273e_coco_20210517_213542-4bc34944.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_fp16_mstrain-608_273e_coco/yolov3_d53_fp16_mstrain-608_273e_coco_20210517_213542.log.json) |

## Lightweight models

|    Backbone     |  Scale  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config   | Download   |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|   MobileNetV2   |   416   |   300e  |   5.3    |                |  23.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_mobilenetv2_mstrain-416_300e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_mobilenetv2_mstrain-416_300e_coco/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_mobilenetv2_mstrain-416_300e_coco/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823.log.json) |
|   MobileNetV2   |   320   |   300e  |   3.2    |                |  22.2  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_mobilenetv2_320_300e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_mobilenetv2_320_300e_coco/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_mobilenetv2_320_300e_coco/yolov3_mobilenetv2_320_300e_coco_20210719_215349.log.json) |

Notice: We reduce the number of channels to 96 in both head and neck. It can reduce the flops and parameters, which makes these models more suitable for edge devices.

## Credit

This implementation originates from the project of Haoyu Wu(@wuhy08) at Western Digital.
