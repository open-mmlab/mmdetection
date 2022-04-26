# RPN

> [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

<!-- [ALGORITHM] -->

## Abstract

State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features---using the recently popular terminology of neural networks with 'attention' mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model, our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143973617-387c7561-82f4-40b2-b78e-4776394b1b8b.png" height="300"/>
</div>

## Results and Models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | AR1000 | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-FPN     |  caffe  |   1x    |   3.5    |      22.6      |  58.7  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_r50_caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_caffe_fpn_1x_coco/rpn_r50_caffe_fpn_1x_coco_20200531-5b903a37.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_caffe_fpn_1x_coco/rpn_r50_caffe_fpn_1x_coco_20200531_012334.log.json) |
|    R-50-FPN     | pytorch |   1x    |   3.8    |      22.3      |  58.2  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_1x_coco/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_1x_coco/rpn_r50_fpn_1x_coco_20200218_151240.log.json) |
|    R-50-FPN     | pytorch |   2x    |    -     |       -        |  58.6  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_r50_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_2x_coco/rpn_r50_fpn_2x_coco_20200131-0728c9b3.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_2x_coco/rpn_r50_fpn_2x_coco_20200131_190631.log.json) |
|    R-101-FPN    |  caffe  |   1x    |   5.4    |      17.3      |  60.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_r101_caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_caffe_fpn_1x_coco/rpn_r101_caffe_fpn_1x_coco_20200531-0629a2e2.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_caffe_fpn_1x_coco/rpn_r101_caffe_fpn_1x_coco_20200531_012345.log.json) |
|    R-101-FPN    | pytorch |   1x    |   5.8    |      16.5      |  59.7  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_r101_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_fpn_1x_coco/rpn_r101_fpn_1x_coco_20200131-2ace2249.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_fpn_1x_coco/rpn_r101_fpn_1x_coco_20200131_191000.log.json) |
|    R-101-FPN    | pytorch |   2x    |    -     |       -        |  60.2  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_r101_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_fpn_2x_coco/rpn_r101_fpn_2x_coco_20200131-24e3db1a.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_fpn_2x_coco/rpn_r101_fpn_2x_coco_20200131_191106.log.json) |
| X-101-32x4d-FPN | pytorch |   1x    |   7.0    |      13.0      |  60.6  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_x101_32x4d_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_1x_coco/rpn_x101_32x4d_fpn_1x_coco_20200219-b02646c6.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_1x_coco/rpn_x101_32x4d_fpn_1x_coco_20200219_012037.log.json) |
| X-101-32x4d-FPN | pytorch |   2x    |    -     |       -        |  61.1  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_x101_32x4d_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_2x_coco/rpn_x101_32x4d_fpn_2x_coco_20200208-d22bd0bb.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_2x_coco/rpn_x101_32x4d_fpn_2x_coco_20200208_200752.log.json) |
| X-101-64x4d-FPN | pytorch |   1x    |   10.1   |      9.1       |  61.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_x101_64x4d_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_1x_coco/rpn_x101_64x4d_fpn_1x_coco_20200208-cde6f7dd.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_1x_coco/rpn_x101_64x4d_fpn_1x_coco_20200208_200752.log.json) |
| X-101-64x4d-FPN | pytorch |   2x    |    -     |       -        |  61.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_x101_64x4d_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_2x_coco/rpn_x101_64x4d_fpn_2x_coco_20200208-c65f524f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_2x_coco/rpn_x101_64x4d_fpn_2x_coco_20200208_200752.log.json) |

## Citation

```latex
@inproceedings{ren2015faster,
  title={Faster r-cnn: Towards real-time object detection with region proposal networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  booktitle={Advances in neural information processing systems},
  year={2015}
}
```
