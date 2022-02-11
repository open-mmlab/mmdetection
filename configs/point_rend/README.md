# PointRend

> [PointRend: Image Segmentation as Rendering](https://arxiv.org/abs/1912.08193)

<!-- [ALGORITHM] -->

## Abstract

We present a new method for efficient high-quality image segmentation of objects and scenes. By analogizing classical computer graphics methods for efficient rendering with over- and undersampling challenges faced in pixel labeling tasks, we develop a unique perspective of image segmentation as a rendering problem. From this vantage, we present the PointRend (Point-based Rendering) neural network module: a module that performs point-based segmentation predictions at adaptively selected locations based on an iterative subdivision algorithm. PointRend can be flexibly applied to both instance and semantic segmentation tasks by building on top of existing state-of-the-art models. While many concrete implementations of the general idea are possible, we show that a simple design already achieves excellent results. Qualitatively, PointRend outputs crisp object boundaries in regions that are over-smoothed by previous methods. Quantitatively, PointRend yields significant gains on COCO and Cityscapes, for both instance and semantic segmentation. PointRend's efficiency enables output resolutions that are otherwise impractical in terms of memory or computation compared to existing approaches.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143970097-d38b6801-d3c8-468f-b8b0-639be3689907.png"/>
</div>

## Results and Models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|    R-50-FPN     |  caffe  |   1x    | 4.6      |                | 38.4   | 36.3    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco/point_rend_r50_caffe_fpn_mstrain_1x_coco_20200612_161407.log.json) |
|    R-50-FPN     |  caffe  |   3x    | 4.6      |                | 41.0   | 38.0    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco/point_rend_r50_caffe_fpn_mstrain_3x_coco_20200614_002632.log.json) |

Note: All models are trained with multi-scale, the input image shorter side is randomly scaled to one of (640, 672, 704, 736, 768, 800).

## Citation

```latex
@InProceedings{kirillov2019pointrend,
  title={{PointRend}: Image Segmentation as Rendering},
  author={Alexander Kirillov and Yuxin Wu and Kaiming He and Ross Girshick},
  journal={ArXiv:1912.08193},
  year={2019}
}
```
