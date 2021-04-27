# SSD: Single Shot MultiBox Detector

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{Liu_2016,
   title={SSD: Single Shot MultiBox Detector},
   journal={ECCV},
   author={Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
   year={2016},
}
```

## Results and models

| Backbone | Size  | Style | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :------: | :---: | :---: | :-----: | :------: | :------------: | :----: | :------: |  :--------: |
|  VGG16   |  300  | caffe |  120e   |   10.2   |  43.7          |  25.6  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd/ssd300_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20200307_174216.log.json) |
|  VGG16   |  512  | caffe |  120e   |   9.3    |  30.7          |  29.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd/ssd512_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20200308-038c5591.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20200308_134447.log.json) |
