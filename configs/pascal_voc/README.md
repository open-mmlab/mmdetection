# Pascal VOC

> [The Pascal Visual Object Classes (VOC) Challenge](https://link.springer.com/article/10.1007/s11263-009-0275-4)

<!-- [DATASET] -->

## Abstract

The Pascal Visual Object Classes (VOC) challenge is a benchmark in visual object category recognition and detection, providing the vision and machine learning communities with a standard dataset of images and annotation, and standard evaluation procedures. Organised annually from 2005 to present, the challenge and its associated dataset has become accepted as the benchmark for object detection.

This paper describes the dataset and evaluation procedure. We review the state-of-the-art in evaluated methods for both classification and detection, analyse whether the methods are statistically different, what they are learning from the images (e.g. the object or its context), and what the methods find easy or confuse. The paper concludes with lessons learnt in the three year history of the challenge, and proposes directions for future improvement and extension.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143969235-6bb4d665-0470-4bae-825c-492eb4582127.png" height="600"/>
</div>

## Results and Models

|  Architecture   | Backbone |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                               Config                                                               |                                                                                                                                                                                                                                                       Download                                                                                                                                                                                                                                                       |
| :-------------: | :------: | :-----: | :-----: | :------: | :------------: | :----: | :--------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Faster R-CNN C4 |   R-50   |  caffe  |   18k   |          |       -        |  80.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/faster_rcnn_r50_caffe_c4_mstrain_18k_voc0712.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/faster_rcnn_r50_caffe_c4_mstrain_18k_voc0712//home/dong/code_sensetime/2022Q1/mmdetection/work_dirs/prepare_voc/gather/pascal_voc/faster_rcnn_r50_caffe_c4_mstrain_18k_voc0712/faster_rcnn_r50_caffe_c4_mstrain_18k_voc0712_20220314_234327-847a14d2.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/faster_rcnn_r50_caffe_c4_mstrain_18k_voc0712/faster_rcnn_r50_caffe_c4_mstrain_18k_voc0712_20220314_234327.log.json) |
|  Faster R-CNN   |   R-50   | pytorch |   1x    |   2.6    |       -        |  80.4  |        [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py)        |                                                                                              [model](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712.log.json)                                                                                              |
|    Retinanet    |   R-50   | pytorch |   1x    |   2.1    |       -        |  77.3  |         [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/retinanet_r50_fpn_1x_voc0712.py)         |                                                                                                     [model](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/retinanet_r50_fpn_1x_voc0712/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/retinanet_r50_fpn_1x_voc0712/retinanet_r50_fpn_1x_voc0712_20200616_014642.log.json)                                                                                                      |
|     SSD300      |  VGG16   |    -    |  120e   |    -     |       -        |  76.5  |                [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/ssd300_voc0712.py)                |                                                                                                                              [model](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/ssd300_voc0712/ssd300_voc0712_20220320_194658-17edda1b.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/ssd300_voc0712/ssd300_voc0712_20220320_194658.log.json)                                                                                                                              |
|     SSD512      |  VGG16   |    -    |  120e   |    -     |       -        |  79.5  |                [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/ssd512_voc0712.py)                |                                                                                                                              [model](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/ssd512_voc0712/ssd512_voc0712_20220320_194717-03cefefe.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/ssd512_voc0712/ssd512_voc0712_20220320_194717.log.json)                                                                                                                              |

## Citation

```latex
@Article{Everingham10,
   author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
   title = "The Pascal Visual Object Classes (VOC) Challenge",
   journal = "International Journal of Computer Vision",
   volume = "88",
   year = "2010",
   number = "2",
   month = jun,
   pages = "303--338",
}
```
