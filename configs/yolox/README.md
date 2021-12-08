# YOLOX: Exceeding YOLO Series in 2021

## Abstract

<!-- [ABSTRACT] -->

In this report, we present some experienced improvements to YOLO series, forming a new high-performance detector -- YOLOX. We switch the YOLO detector to an anchor-free manner and conduct other advanced detection techniques, i.e., a decoupled head and the leading label assignment strategy SimOTA to achieve state-of-the-art results across a large scale range of models: For YOLO-Nano with only 0.91M parameters and 1.08G FLOPs, we get 25.3% AP on COCO, surpassing NanoDet by 1.8% AP; for YOLOv3, one of the most widely used detectors in industry, we boost it to 47.3% AP on COCO, outperforming the current best practice by 3.0% AP; for YOLOX-L with roughly the same amount of parameters as YOLOv4-CSP, YOLOv5-L, we achieve 50.0% AP on COCO at a speed of 68.9 FPS on Tesla V100, exceeding YOLOv5-L by 1.8% AP. Further, we won the 1st Place on Streaming Perception Challenge (Workshop on Autonomous Driving at CVPR 2021) using a single YOLOX-L model. We hope this report can provide useful experience for developers and researchers in practical scenes, and we also provide deploy versions with ONNX, TensorRT, NCNN, and Openvino supported.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/144001736-9fb303dd-eac7-46b0-ad45-214cfa51e928.png"/>
</div>

<!-- [PAPER_TITLE: YOLOX: Exceeding YOLO Series in 2021] -->
<!-- [PAPER_URL: https://arxiv.org/abs/2107.08430] -->

## Citation

<!-- [ALGORITHM] -->

```latex
@article{yolox2021,
  title={{YOLOX}: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

## Results and Models

| Backbone  | size   | Mem (GB) |   box AP | Config | Download |
|:---------:|:-------:|:-------:|:-------:|:--------:|:------:|
| YOLOX-tiny | 416 |   3.5      |   32.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox/yolox_tiny_8x8_300e_coco.py)       |[model](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234.log.json) |
| YOLOX-s | 640 |   7.6      |   40.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox/yolox_s_8x8_300e_coco.py)       |[model](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711.log.json) |
| YOLOX-l | 640 |   19.9      |   49.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox/yolox_l_8x8_300e_coco.py)       |[model](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236.log.json) |
| YOLOX-x | 640 |   28.1      |   50.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox/yolox_x_8x8_300e_coco.py)       |[model](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254.log.json) |


**Note**:

1. The test score threshold is 0.001, and the box AP indicates the best AP.
2. Due to the need for pre-training weights, we cannot reproduce the performance of the `yolox-nano` model. Please refer to https://github.com/Megvii-BaseDetection/YOLOX/issues/674 for more information.
3. We also trained the model by the official release of YOLOX based on [Megvii-BaseDetection/YOLOX#735](https://github.com/Megvii-BaseDetection/YOLOX/issues/735) with commit ID [38c633](https://github.com/Megvii-BaseDetection/YOLOX/tree/38c633bf176462ee42b110c70e4ffe17b5753208). We found that the best AP of `YOLOX-tiny`, `YOLOX-s`, `YOLOX-l`, and `YOLOX-x` is 31.8, 40.3, 49.2, and 50.9, respectively. The performance is consistent with that of our re-implementation (see Table above) but still has a gap (0.3~0.8 AP) in comparison with the reported performance in their [README](https://github.com/Megvii-BaseDetection/YOLOX/blob/38c633bf176462ee42b110c70e4ffe17b5753208/README.md#benchmark).
