# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

## Introduction
```
@inproceedings{ren2015faster,
  title={Faster r-cnn: Towards real-time object detection with region proposal networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  booktitle={Advances in neural information processing systems},
  year={2015}
}
```

## Results and models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | AR1000 | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :--------: |
|    R-50-FPN     |  caffe  |   1x    |          |                |        | |
|    R-50-FPN     | pytorch |   1x    |   3.8    |      19.7      |  58.2  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_r50_fpn_1x_coco/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_r50_fpn_1x_coco/rpn_r50_fpn_1x_coco_20200218_151240.log.json) |
|    R-50-FPN     | pytorch |   2x    |    -     |       -        |  58.6  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_r50_fpn_2x_coco/rpn_r50_fpn_2x_coco_20200131-0728c9b3.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_r50_fpn_2x_coco/rpn_r50_fpn_2x_coco_20200131_190631.log.json) |
|    R-101-FPN    |  caffe  |   1x    |          |                |        | |
|    R-101-FPN    | pytorch |   1x    |   5.8    |      15.0      |  59.7  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_r101_fpn_1x_coco/rpn_r101_fpn_1x_coco_20200131-2ace2249.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_r101_fpn_1x_coco/rpn_r101_fpn_1x_coco_20200131_191000.log.json) |
|    R-101-FPN    | pytorch |   2x    |    -     |       -        |  60.2  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_r101_fpn_2x_coco/rpn_r101_fpn_2x_coco_20200131-24e3db1a.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_r101_fpn_2x_coco/rpn_r101_fpn_2x_coco_20200131_191106.log.json) |
| X-101-32x4d-FPN | pytorch |   1x    |   7.0    |      12.3      |  60.6  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_1x_coco/rpn_x101_32x4d_fpn_1x_coco_20200219-b02646c6.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_1x_coco/rpn_x101_32x4d_fpn_1x_coco_20200219_012037.log.json) |
| X-101-32x4d-FPN | pytorch |   2x    |    -     |       -        |  61.1  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_2x_coco/rpn_x101_32x4d_fpn_2x_coco_20200208-d22bd0bb.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_2x_coco/rpn_x101_32x4d_fpn_2x_coco_20200208_200752.log.json) |
| X-101-64x4d-FPN | pytorch |   1x    |   10.1   |      8.7       |  61.0  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_1x_coco/rpn_x101_64x4d_fpn_1x_coco_20200208-cde6f7dd.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_1x_coco/rpn_x101_64x4d_fpn_1x_coco_20200208_200752.log.json) |
| X-101-64x4d-FPN | pytorch |   2x    |    -     |       -        |  61.5  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_2x_coco/rpn_x101_64x4d_fpn_2x_coco_20200208-c65f524f.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_2x_coco/rpn_x101_64x4d_fpn_2x_coco_20200208_200752.log.json) |
