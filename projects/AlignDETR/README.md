# AlignDETR

> [Align-DETR: Improving DETR with Simple IoU-aware BCE loss](https://arxiv.org/abs/2304.07527)

<!-- [ALGORITHM] -->

## Abstract

DETR has set up a simple end-to-end pipeline for object detection by formulating this task as a set prediction problem, showing promising potential. However, despite the significant progress in improving DETR, this paper identifies a problem of misalignment in the output distribution, which prevents the best-regressed samples from being assigned with high confidence, hindering the model's accuracy. We propose a metric, recall of best-regressed samples, to quantitively evaluate the misalignment problem. Observing its importance, we propose a novel Align-DETR that incorporates a localization precision-aware classification loss in optimization. The proposed loss, IA-BCE, guides the training of DETR to build a strong correlation between classification score and localization precision. We also adopt the mixed-matching strategy, to facilitate DETR-based detectors with faster training convergence while keeping an end-to-end scheme. Moreover, to overcome the dramatic decrease in sample quality induced by the sparsity of queries, we introduce a prime sample weighting mechanism to suppress the interference of unimportant samples. Extensive experiments are conducted with very competitive results reported. In particular, it delivers a 46 (+3.8)% AP on the DAB-DETR baseline with the ResNet-50 backbone and reaches a new SOTA performance of 50.2% AP in the 1x setting on the COCO validation set when employing the strong baseline DINO.

![image](https://github.com/open-mmlab/mmdetection/assets/33146359/5a4fa664-b4c6-487d-b6d8-22be9d59a2bc)

## Results and Models

| Backbone |    Model    | Lr schd | box AP |                       Config                       |                                                                                                                                                                    Download                                                                                                                                                                    |
| :------: | :---------: | :-----: | :----: | :------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | DINO-4scale |   12e   |  50.5  | [config](./align_detr-4scale_r50_8xb2-12e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/align_detr/align_detr-4scale_r50_8xb2-12e_coco/align_detr-4scale_r50_8xb2-12e_coco_20230914_095734-61f921af.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/align_detr/align_detr-4scale_r50_8xb2-12e_coco/align_detr-4scale_r50_8xb2-12e_coco_20230914_095734.log.json) |
|   R-50   | DINO-4scale |   24e   |  51.4  | [config](./align_detr-4scale_r50_8xb2-24e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/align_detr/align_detr-4scale_r50_8xb2-24e_coco/align_detr-4scale_r50_8xb2-24e_coco_20230919_152414-f4b6cf76.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/align_detr/align_detr-4scale_r50_8xb2-24e_coco/align_detr-4scale_r50_8xb2-24e_coco_20230919_152414.log.json) |

## Citation

We provide the config files for AlignDETR: [Align-DETR: Improving DETR with Simple IoU-aware BCE loss](https://arxiv.org/abs/2304.07527).

```latex
@misc{cai2023aligndetr,
      title={Align-DETR: Improving DETR with Simple IoU-aware BCE loss},
      author={Zhi Cai and Songtao Liu and Guodong Wang and Zheng Ge and Xiangyu Zhang and Di Huang},
      year={2023},
      eprint={2304.07527},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
