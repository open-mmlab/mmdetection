# Scale-Aware Trident Networks for Object Detection

## Introduction

[ALGORITHM]

```
@InProceedings{li2019scale,
  title={Scale-Aware Trident Networks for Object Detection},
  author={Li, Yanghao and Chen, Yuntao and Wang, Naiyan and Zhang, Zhaoxiang},
  journal={The International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

## Results and models

We reports the test results using only one branch for inference.

|    Backbone     |  Style  | mstrain | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
| :-------------: | :-----: | :-----: | :-----: | :------: | :------------: | :----: | :------: |
|    R-50         |  caffe  |    N    |   1x    |          |                | 37.7   |[model](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_1x_coco/tridentnet_r50_caffe_1x_coco_20201230_141838-2ec0b530.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_1x_coco/tridentnet_r50_caffe_1x_coco_20201230_141838.log.json) |
|    R-50         |  caffe  |    Y    |   1x    |          |                | 37.6   |[model](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_mstrain_1x_coco/tridentnet_r50_caffe_mstrain_1x_coco_20201230_141839-6ce55ccb.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_mstrain_1x_coco/tridentnet_r50_caffe_mstrain_1x_coco_20201230_141839.log.json) |
|    R-50         |  caffe  |    Y    |   3x    |          |                | 40.3   |[model](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_mstrain_3x_coco/tridentnet_r50_caffe_mstrain_3x_coco_20201130_100539-46d227ba.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_mstrain_3x_coco/tridentnet_r50_caffe_mstrain_3x_coco_20201130_100539.log.json) |
