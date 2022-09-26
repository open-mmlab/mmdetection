# Double Heads

> [Rethinking Classification and Localization for Object Detection](https://arxiv.org/abs/1904.06493)

<!-- [ALGORITHM] -->

## Abstract

Two head structures (i.e. fully connected head and convolution head) have been widely used in R-CNN based detectors for classification and localization tasks. However, there is a lack of understanding of how does these two head structures work for these two tasks. To address this issue, we perform a thorough analysis and find an interesting fact that the two head structures have opposite preferences towards the two tasks. Specifically, the fully connected head (fc-head) is more suitable for the classification task, while the convolution head (conv-head) is more suitable for the localization task. Furthermore, we examine the output feature maps of both heads and find that fc-head has more spatial sensitivity than conv-head. Thus, fc-head has more capability to distinguish a complete object from part of an object, but is not robust to regress the whole object. Based upon these findings, we propose a Double-Head method, which has a fully connected head focusing on classification and a convolution head for bounding box regression. Without bells and whistles, our method gains +3.5 and +2.8 AP on MS COCO dataset from Feature Pyramid Network (FPN) baselines with ResNet-50 and ResNet-101 backbones, respectively.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143879010-e30f654b-f93e-44b2-a186-c251fdca5bda.png"/>
</div>

## Results and Models

| Backbone |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                    Config                     |                                                                                                                                                        Download                                                                                                                                                         |
| :------: | :-----: | :-----: | :------: | :------------: | :----: | :-------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| R-50-FPN | pytorch |   1x    |   6.8    |      9.5       |  40.0  | [config](./dh-faster-rcnn_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/double_heads/dh_faster_rcnn_r50_fpn_1x_coco/dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/double_heads/dh_faster_rcnn_r50_fpn_1x_coco/dh_faster_rcnn_r50_fpn_1x_coco_20200130_220238.log.json) |

## Citation

```latex
@article{wu2019rethinking,
    title={Rethinking Classification and Localization for Object Detection},
    author={Yue Wu and Yinpeng Chen and Lu Yuan and Zicheng Liu and Lijuan Wang and Hongzhi Li and Yun Fu},
    year={2019},
    eprint={1904.06493},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
