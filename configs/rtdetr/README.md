# RT-DETR

> [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)

<!-- [ALGORITHM] -->

## Abstract

Recently, end-to-end transformer-based detectors (DETRs) have achieved remarkable performance. However, the issue of the high computational cost of DETRs has not been effectively addressed, limiting their practical application and preventing them from fully exploiting the benefits of no post-processing, such as non-maximum suppression (NMS). In this paper, we first analyze the influence of NMS in modern real-time object detectors on inference speed, and establish an end-to-end speed benchmark. To avoid the inference delay caused by NMS, we propose a Real-Time DEtection TRansformer (RT-DETR), the first real-time end-to-end object detector to our best knowledge. Specifically, we design an efficient hybrid encoder to efficiently process multi-scale features by decoupling the intra-scale interaction and cross-scale fusion, and propose IoU-aware query selection to improve the initialization of object queries. In addition, our proposed detector supports flexibly adjustment of the inference speed by using different decoder layers without the need for retraining, which facilitates the practical application of real-time object detectors. Our RT-DETR-L achieves 53.0% AP on COCO val2017 and 114 FPS on T4 GPU, while RT-DETR-X achieves 54.8% AP and 74 FPS, outperforming all YOLO detectors of the same scale in both speed and accuracy. Furthermore, our RT-DETR-R50 achieves 53.1% AP and 108 FPS, outperforming DINO-Deformable-DETR-R50 by 2.2% AP in accuracy and by about 21 times in FPS. Source code and pretrained models will be available at PaddleDetection.

<div align=center>
<img src="https://user-images.githubusercontent.com/17582080/245363952-196b0a10-d2e8-401c-9132-54b9126e0a33.png"/>
</div>

## Results and Models

| Backbone |     Model     | Lr schd | box AP |                  Config                   |                                               Download                                                |
| :------: | :-----------: | :-----: | :----: | :---------------------------------------: | :---------------------------------------------------------------------------------------------------: |
|   R-50   | RT-DETR-R50\* |   72e   |  53.1  | [config](./rtdetr_r50vd_8xb2-72e_coco.py) | [model](https://github.com/nijkah/storage/releases/download/v0.0.1/rtdetr_r50vd_6x_coco_mmdet.pth) \| |

### NOTE

Models with * are converted from the [official repo](https://github.com/PaddlePaddle/PaddleDetection/). The config files of these models are only for inference. We haven't reprodcue the training results.

## Citation

```latex
@article{lv2023detrs,
  title={Detrs beat yolos on real-time object detection},
  author={Lv, Wenyu and Xu, Shangliang and Zhao, Yian and Wang, Guanzhong and Wei, Jinman and Cui, Cheng and Du, Yuning and Dang, Qingqing and Liu, Yi},
  journal={arXiv preprint arXiv:2304.08069},
  year={2023}
}
```
