# DDOD

> [Disentangle Your Dense Object Detector](https://arxiv.org/pdf/2107.02963.pdf)

<!-- [DATASET] -->

## Abstract

Deep learning-based dense object detectors have achieved great success in the past few years and have been applied to numerous multimedia applications such as video understanding. However, the current training pipeline for dense detectors is compromised to lots of conjunctions that may not hold. In this paper, we investigate three such important conjunctions: 1) only samples assigned as positive in classification head are used to train the regression head; 2) classification and regression share the same input feature and computational fields defined by the parallel head architecture; and 3) samples distributed in different feature pyramid layers are treated equally when computing the loss. We first carry out a series of pilot experiments to show disentangling such conjunctions can lead to persistent performance improvement. Then, based on these findings, we propose Disentangled Dense Object Detector(DDOD), in which simple and effective disentanglement mechanisms are designed and integrated into the current state-of-the-art dense object detectors. Extensive experiments on MS COCO benchmark show that our approach can lead to 2.0 mAP, 2.4 mAP and 2.2 mAP absolute improvements on RetinaNet, FCOS, and ATSS baselines with negligible extra overhead. Notably, our best model reaches 55.0 mAP on the COCO test-dev set and 93.5 AP on the hard subset of WIDER FACE, achieving new state-of-the-art performance on these two competitive benchmarks. Code is available at https://github.com/zehuichen123/DDOD.

## Results and Models

| Model | Backbone | Lr Schd | box mAP | AP50 | AP75 | APs | APm | APl |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| ATSS(IoU) | ResNet50 | 1x | 39.4 | 56.6 | 42.6 | 23.9 | 42.5 | 49.6 |
| DDOD | ResNet50 | 1x | 41.6 | 59.9 | 45.2 | 23.9 | 44.9 | 54.4 |
| DDOD-FCOS | ResNet50 | 1x | 41.6 | 59.9 | 45.3 | 24.0 | 44.6 | 54.8 |


## Citing
```
@misc{chen2021disentangle,
      title={Disentangle Your Dense Object Detector}, 
      author={Zehui Chen and Chenhongyi Yang and Qiaofei Li and Feng Zhao and Zhengjun Zha and Feng Wu},
      year={2021},
      eprint={2107.02963},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
