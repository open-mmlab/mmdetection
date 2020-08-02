# YOLOv3

## Introduction
```
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

|    Backbone     |  Scale  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download  |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-------: |
|   DarkNet-53    |   320   |   273e  |   2.7    |      50.9      |  28.6  | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-7cf15c63.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-20200801_151445.log.json) |
|   DarkNet-53    |   416   |   273e  |   3.8    |      50.9      |  30.7  | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-1a24e53b.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-20200801_150219.log.json) |
|   DarkNet-53    |   608   |   273e  |   7.1    |      47.1      |  33.6  | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco-d97b9e5c.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco-20200801_150110.log.json) |


## Credit
This implementation originates from the project of Haoyu Wu(@wuhy08) at Western Digital.
