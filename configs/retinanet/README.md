# Focal Loss for Dense Object Detection

## Introduction

[ALGORITHM]

```latex
@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```

## Results and models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download  |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-FPN     |  caffe  |   1x    |   3.5    |      18.6      |  36.3  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_caffe_fpn_1x_coco/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_caffe_fpn_1x_coco/retinanet_r50_caffe_fpn_1x_coco_20200531_012518.log.json) |
|    R-50-FPN     | pytorch |   1x    |   3.8    |      19.0      |  36.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130_002941.log.json) |
|    R-50-FPN     | pytorch |   2x    |    -     |       -        |  37.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131_114738.log.json) |
|    R-101-FPN    |  caffe  |   1x    |   5.5    |      14.7      |  38.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r101_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_caffe_fpn_1x_coco/retinanet_r101_caffe_fpn_1x_coco_20200531-b428fa0f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_caffe_fpn_1x_coco/retinanet_r101_caffe_fpn_1x_coco_20200531_012536.log.json) |
|    R-101-FPN    | pytorch |   1x    |   5.7    |      15.0      |  38.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_1x_coco/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_1x_coco/retinanet_r101_fpn_1x_coco_20200130_003055.log.json) |
|    R-101-FPN    | pytorch |   2x    |    -     |       -        |  38.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r101_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131_114859.log.json) |
| X-101-32x4d-FPN | pytorch |   1x    |   7.0    |      12.1      |  39.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_1x_coco/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_1x_coco/retinanet_x101_32x4d_fpn_1x_coco_20200130_003004.log.json) |
| X-101-32x4d-FPN | pytorch |   2x    |    -     |       -        |  40.1  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_32x4d_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_2x_coco/retinanet_x101_32x4d_fpn_2x_coco_20200131-237fc5e1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_2x_coco/retinanet_x101_32x4d_fpn_2x_coco_20200131_114812.log.json) |
| X-101-64x4d-FPN | pytorch |   1x    |   10.0   |      8.7       |  41.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130_003008.log.json) |
| X-101-64x4d-FPN | pytorch |   2x    |    -     |       -        |  40.8  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_64x4d_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_2x_coco/retinanet_x101_64x4d_fpn_2x_coco_20200131-bca068ab.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_2x_coco/retinanet_x101_64x4d_fpn_2x_coco_20200131_114833.log.json) |
