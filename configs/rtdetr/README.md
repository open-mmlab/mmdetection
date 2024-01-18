# RT-DETR

> [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)

<!-- [ALGORITHM] -->

## Abstract

Recently, end-to-end transformer-based detectors~(DETRs) have achieved remarkable performance. However, the issue of the high computational cost of DETRs has not been effectively addressed, limiting their practical application and preventing them from fully exploiting the benefits of no post-processing, such as non-maximum suppression (NMS). In this paper, we first analyze the influence of NMS in modern real-time object detectors on inference speed, and establish an end-to-end speed benchmark. To avoid the inference delay caused by NMS, we propose a Real-Time DEtection TRansformer (RT-DETR), the first real-time end-to-end object detector to our best knowledge. Specifically, we design an efficient hybrid encoder to efficiently process multi-scale features by decoupling the intra-scale interaction and cross-scale fusion, and propose IoU-aware query selection to improve the initialization of object queries. In addition, our proposed detector supports flexibly adjustment of the inference speed by using different decoder layers without the need for retraining, which facilitates the practical application of real-time object detectors. Our RT-DETR-L achieves 53.0% AP on COCO val2017 and 114 FPS on T4 GPU, while RT-DETR-X achieves 54.8% AP and 74 FPS, outperforming all YOLO detectors of the same scale in both speed and accuracy. Furthermore, our RT-DETR-R50 achieves 53.1% AP and 108 FPS, outperforming DINO-Deformable-DETR-R50 by 2.2% AP in accuracy and by about 21 times in FPS. ource code and pre-trained models are available at [this https URL](https://github.com/lyuwenyu/RT-DETR).

<div align=center>
<img src="https://user-images.githubusercontent.com/17582080/262603054-42636690-1ecf-4647-b075-842ecb9bc562.png"/>
</div>

## Results and Models

| Backbone |    Model     | Lr schd | box AP |                   Config                   |                                                          Download                                                           |
| :------: | :----------: | :-----: | :----: | :----------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
|  R-18vd  | RT-DETR Dec3 |   72e   |  46.5  | [config](./rtdetr_r18vd_8xb2-72e_coco.py)  | [model](https://github.com/flytocc/mmdetection/releases/download/model_zoo/rtdetr_r18vd_8xb2-72e_coco_d214e55d.pth)  \| log |
|  R-34vd  | RT-DETR Dec4 |   72e   |  48.9  | [config](./rtdetr_r34vd_8xb2-72e_coco.py)  | [model](https://github.com/flytocc/mmdetection/releases/download/model_zoo/rtdetr_r34vd_8xb2-72e_coco_4c1cbe01.pth)  \| log |
|  R-50vd  | RT-DETR Dec6 |   72e   |  53.1  | [config](./rtdetr_r50vd_8xb2-72e_coco.py)  | [model](https://github.com/flytocc/mmdetection/releases/download/model_zoo/rtdetr_r50vd_8xb2-72e_coco_ff87da1a.pth)  \| log |
| R-101vd  | RT-DETR Dec6 |   72e   |  54.3  | [config](./rtdetr_r101vd_8xb2-72e_coco.py) | [model](https://github.com/flytocc/mmdetection/releases/download/model_zoo/rtdetr_r101vd_8xb2-72e_coco_104a0e6b.pth) \| log |

### NOTE

Weights converted from the [official repo](https://github.com/lyuwenyu/RT-DETR).

The performance is unstable. `RT-DETR` with `R-50vd` may fluctuate about 0.4 mAP.

## Citation

We provide the config files for RT-DETR: [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069).

```latex
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
