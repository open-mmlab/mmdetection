# The Pascal Visual Object Classes (VOC) Challenge

## Abstract

<!-- [ABSTRACT] -->

The Pascal Visual Object Classes (VOC) challenge is a benchmark in visual object category recognition and detection, providing the vision and machine learning communities with a standard dataset of images and annotation, and standard evaluation procedures. Organised annually from 2005 to present, the challenge and its associated dataset has become accepted as the benchmark for object detection.

This paper describes the dataset and evaluation procedure. We review the state-of-the-art in evaluated methods for both classification and detection, analyse whether the methods are statistically different, what they are learning from the images (e.g. the object or its context), and what the methods find easy or confuse. The paper concludes with lessons learnt in the three year history of the challenge, and proposes directions for future improvement and extension.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143969235-6bb4d665-0470-4bae-825c-492eb4582127.png" height="600"/>
</div>

<!-- [PAPER_TITLE: The Pascal Visual Object Classes (VOC) Challenge] -->
<!-- [PAPER_URL: https://link.springer.com/article/10.1007/s11263-009-0275-4] -->

## Citation

<!-- [DATASET] -->

```
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

## Results and Models

| Architecture | Backbone  | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:------------:|:---------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| Faster R-CNN | R-50      | pytorch | 1x      | 2.6   | -          | 79.5  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712/faster_rcnn_r50_fpn_1x_voc0712_20200624-c9895d40.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712/20200623_015208.log.json) |
| Retinanet    | R-50      | pytorch | 1x      | 2.1   | -          | 77.3  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/retinanet_r50_fpn_1x_voc0712.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/retinanet_r50_fpn_1x_voc0712/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/retinanet_r50_fpn_1x_voc0712/retinanet_r50_fpn_1x_voc0712_20200616_014642.log.json) |
