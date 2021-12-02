# YOLOv3: An Incremental Improvement

## Abstract

<!-- [ABSTRACT] -->

We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that's pretty swell. It's a little bigger than last time but more accurate. It's still fast though, don't worry. At 320x320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 mAP@50 in 51 ms on a Titan X, compared to 57.5 mAP@50 in 198 ms by RetinaNet, similar performance but 3.8x faster.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/144001433-b4f7fb5e-3b7a-414b-b949-93733213b670.png" height="300"/>
</div>

<!-- [PAPER_TITLE: YOLOv3: An Incremental Improvement] -->
<!-- [PAPER_URL: https://arxiv.org/abs/1804.02767] -->

## Citation

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
