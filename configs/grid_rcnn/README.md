# Grid R-CNN

> [Grid R-CNN](https://arxiv.org/abs/1811.12030)

<!-- [ALGORITHM] -->

## Abstract

This paper proposes a novel object detection framework named Grid R-CNN, which adopts a grid guided localization mechanism for accurate object detection. Different from the traditional regression based methods, the Grid R-CNN captures the spatial information explicitly and enjoys the position sensitive property of fully convolutional architecture. Instead of using only two independent points, we design a multi-point supervision formulation to encode more clues in order to reduce the impact of inaccurate prediction of specific points. To take the full advantage of the correlation of points in a grid, we propose a two-stage information fusion strategy to fuse feature maps of neighbor grid points. The grid guided localization approach is easy to be extended to different state-of-the-art detection frameworks. Grid R-CNN leads to high quality object localization, and experiments demonstrate that it achieves a 4.1% AP gain at IoU=0.8 and a 10.0% AP gain at IoU=0.9 on COCO benchmark compared to Faster R-CNN with Res50 backbone and FPN architecture.

Grid R-CNN is a well-performed objection detection framework. It transforms the traditional box offset regression problem into a grid point estimation problem. With the guidance of the grid points, it can obtain high-quality localization results. However, the speed of Grid R-CNN is not so satisfactory. In this technical report we present Grid R-CNN Plus, a better and faster version of Grid R-CNN. We have made several updates that significantly speed up the framework and simultaneously improve the accuracy. On COCO dataset, the Res50-FPN based Grid R-CNN Plus detector achieves an mAP of 40.4%, outperforming the baseline on the same model by 3.0 points with similar inference time.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143890379-5d9e6233-0533-48b4-88b9-bc33abbd9f82.png"/>
</div>

## Results and Models

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                            Config                                                             |                                                                                                                                                                         Download                                                                                                                                                                          |
| :---------: | :-----: | :------: | :------------: | :----: | :---------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50     |   2x    |   5.1    |      15.0      |  40.4  |    [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py)     |               [model](https://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130_221140.log.json)               |
|    R-101    |   2x    |   7.0    |      12.6      |  41.5  |    [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/grid_rcnn/grid_rcnn_r101_fpn_gn-head_2x_coco.py)    |             [model](https://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r101_fpn_gn-head_2x_coco/grid_rcnn_r101_fpn_gn-head_2x_coco_20200309-d6eca030.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r101_fpn_gn-head_2x_coco/grid_rcnn_r101_fpn_gn-head_2x_coco_20200309_164224.log.json)             |
| X-101-32x4d |   2x    |   8.3    |      10.8      |  42.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/grid_rcnn/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco_20200130-d8f0e3ff.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco_20200130_215413.log.json) |
| X-101-64x4d |   2x    |   11.3   |      7.7       |  43.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco_20200204-ec76a754.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco_20200204_080641.log.json) |

**Notes:**

- All models are trained with 8 GPUs instead of 32 GPUs in the original paper.
- The warming up lasts for 1 epoch and `2x` here indicates 25 epochs.

## Citation

```latex
@inproceedings{lu2019grid,
  title={Grid r-cnn},
  author={Lu, Xin and Li, Buyu and Yue, Yuxin and Li, Quanquan and Yan, Junjie},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}

@article{lu2019grid,
  title={Grid R-CNN Plus: Faster and Better},
  author={Lu, Xin and Li, Buyu and Yue, Yuxin and Li, Quanquan and Yan, Junjie},
  journal={arXiv preprint arXiv:1906.05688},
  year={2019}
}
```
