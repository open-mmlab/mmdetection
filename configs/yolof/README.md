# You Only Look One-level Feature

## Abstract

<!-- [ABSTRACT] -->

This paper revisits feature pyramids networks (FPN) for one-stage detectors and points out that the success of FPN is due to its divide-and-conquer solution to the optimization problem in object detection rather than multi-scale feature fusion. From the perspective of optimization, we introduce an alternative way to address the problem instead of adopting the complex feature pyramids - {\em utilizing only one-level feature for detection}. Based on the simple and efficient solution, we present You Only Look One-level Feature (YOLOF). In our method, two key components, Dilated Encoder and Uniform Matching, are proposed and bring considerable improvements. Extensive experiments on the COCO benchmark prove the effectiveness of the proposed model. Our YOLOF achieves comparable results with its feature pyramids counterpart RetinaNet while being 2.5× faster. Without transformer layers, YOLOF can match the performance of DETR in a single-level feature manner with 7× less training epochs. With an image size of 608×608, YOLOF achieves 44.3 mAP running at 60 fps on 2080Ti, which is 13% faster than YOLOv4.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/144001639-257374ef-7d4f-412b-a783-88abdd22f277.png"/>
</div>

<!-- [PAPER_TITLE: You Only Look One-level Feature] -->
<!-- [PAPER_URL: https://arxiv.org/abs/2103.09460] -->

## Citation

<!-- [ALGORITHM] -->

```
@inproceedings{chen2021you,
  title={You Only Look One-level Feature},
  author={Chen, Qiang and Wang, Yingming and Yang, Tong and Zhang, Xiangyu and Cheng, Jian and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

## Results and Models

| Backbone  | Style   | Epoch | Lr schd | Mem (GB) |   box AP | Config | Download |
|:---------:|:-------:|:-------:|:-------:|:--------:|:------:|:------:|:--------:|
| R-50-C5     | caffe | Y | 1x      | 8.3      |   37.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolof/yolof_r50_c5_8x8_1x_coco.py)       |[model](https://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427.log.json) |

**Note**:

1. We find that the performance is unstable and may fluctuate by about 0.3 mAP. mAP 37.4 ~ 37.7 is acceptable in YOLOF_R_50_C5_1x. Such fluctuation can also be found in the [original implementation](https://github.com/chensnathan/YOLOF).
2. In addition to instability issues, sometimes there are large loss fluctuations and NAN, so there may still be problems with this project, which will be improved subsequently.
