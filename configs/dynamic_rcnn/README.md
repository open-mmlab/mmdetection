# Dynamic R-CNN

> [Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training](https://arxiv.org/abs/2004.06002)

<!-- [ALGORITHM] -->

## Abstract

Although two-stage object detectors have continuously advanced the state-of-the-art performance in recent years, the training process itself is far from crystal. In this work, we first point out the inconsistency problem between the fixed network settings and the dynamic training procedure, which greatly affects the performance. For example, the fixed label assignment strategy and regression loss function cannot fit the distribution change of proposals and thus are harmful to training high quality detectors. Consequently, we propose Dynamic R-CNN to adjust the label assignment criteria (IoU threshold) and the shape of regression loss function (parameters of SmoothL1 Loss) automatically based on the statistics of proposals during training. This dynamic design makes better use of the training samples and pushes the detector to fit more high quality samples. Specifically, our method improves upon ResNet-50-FPN baseline with 1.9% AP and 5.5% AP90 on the MS COCO dataset with no extra overhead.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143879518-842f5bec-9f65-4454-93a1-9b3b0c42ec3c.png"/>
</div>

## Results and Models

| Backbone  | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50      | pytorch | 1x      | 3.8      |                |  38.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x/dynamic_rcnn_r50_fpn_1x-62a3f276.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x/dynamic_rcnn_r50_fpn_1x_20200618_095048.log.json) |

## Citation

```latex
@article{DynamicRCNN,
    author = {Hongkai Zhang and Hong Chang and Bingpeng Ma and Naiyan Wang and Xilin Chen},
    title = {Dynamic {R-CNN}: Towards High Quality Object Detection via Dynamic Training},
    journal = {arXiv preprint arXiv:2004.06002},
    year = {2020}
}
```
