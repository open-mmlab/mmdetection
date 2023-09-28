# H-DETR

> [DETRs with Hybrid Matching](https://arxiv.org/abs/2207.13080)

<!-- [ALGORITHM] -->

## Abstract

One-to-one set matching is a key design for DETR to establish its end-to-end capability, so that object detection does not require a hand-crafted NMS (non-maximum suppression) to remove duplicate detections. This end-to-end signature is important for the versatility of DETR, and it has been generalized to broader vision tasks. However, we note that there are few queries assigned as positive samples and the one-to-one set matching significantly reduces the training efficacy of positive samples. We propose a simple yet effective method based on a hybrid matching scheme that combines the original one-to-one matching branch with an auxiliary one-to-many matching branch during training. Our hybrid strategy has been shown to significantly improve accuracy. In inference, only the original one-to-one match branch is used, thus maintaining the end-to-end merit and the same inference efficiency of DETR. The method is named H-DETR, and it shows that a wide range of representative DETR methods can be consistently improved across a wide range of visual tasks, including DeformableDETR, PETRv2, PETR, and TransTrack, among others.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/254f3037-1ca8-4d0c-8f3e-45d8ec3f9abc"/>
</div>

## Results and Models

| Backbone |     Model     | Lr schd | box AP |                     Config                     | Download |
| :------: | :-----------: | :-----: | :----: | :--------------------------------------------: | :------: |
|   R-50   | H-DINO-4scale |   12e   |  48.0  | [config](./h-dino-4scale_r50_8xb2-12e_coco.py) |          |

### NOTE

1. We are based on `DINO` rather than `Deformable DETR` to support the `Hybrid Matching` algorithm.
2. We found that directly applying Hybrid Matching to the DINO algorithm results in a significant decrease in performance. If you have any other insights or suggestions, please feel free to comment or submit a pull request (PR).

## Citation

```latex
@article{jia2022detrs,
  title={DETRs with Hybrid Matching},
  author={Jia, Ding and Yuan, Yuhui and He, Haodi and Wu, Xiaopei and Yu, Haojun and Lin, Weihong and Sun, Lei and Zhang, Chao and Hu, Han},
  journal={arXiv preprint arXiv:2207.13080},
  year={2022}
}
```
