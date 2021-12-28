# Objects as Points

## Abstract

<!-- [ABSTRACT] -->

Detection identifies objects as axis-aligned boxes in an image. Most successful object detectors enumerate a nearly exhaustive list of potential object locations and classify each. This is wasteful, inefficient, and requires additional post-processing. In this paper, we take a different approach. We model an object as a single point --- the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. Our center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors. CenterNet achieves the best speed-accuracy trade-off on the MS COCO dataset, with 28.1% AP at 142 FPS, 37.4% AP at 52 FPS, and 45.1% AP with multi-scale testing at 1.4 FPS. We use the same approach to estimate 3D bounding box in the KITTI benchmark and human pose on the COCO keypoint dataset. Our method performs competitively with sophisticated multi-stage methods and runs in real-time.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143873810-85ffa6e7-915b-46a4-9b8f-709e5d7700bb.png"/>
</div>

<!-- [PAPER_TITLE: Objects as Points] -->
<!-- [PAPER_URL: https://arxiv.org/abs/1904.07850] -->

## Citation

<!-- [ALGORITHM] -->

```latex
@article{zhou2019objects,
  title={Objects as Points},
  author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle={arXiv preprint arXiv:1904.07850},
  year={2019}
}
```

## Results and models

| Backbone        | DCN |  Mem (GB) | Box AP | Flip box AP| Config | Download |
| :-------------: | :--------: |:----------------: | :------: | :------------: | :----: | :----: |
| ResNet-18 | N | 3.45 | 25.9 | 27.3 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/centernet/centernet_resnet18_140e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210705_093630.log.json) |
| ResNet-18 | Y | 3.47 | 29.5 | 30.9 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/centernet/centernet_resnet18_dcnv2_140e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco_20210702_155131.log.json) |

Note:

- Flip box AP setting is single-scale and `flip=True`.
- Due to complex data enhancement, we find that the performance is unstable and may fluctuate by about 0.4 mAP. mAP 29.4 ~ 29.8 is acceptable in ResNet-18-DCNv2.
- Compared to the source code, we refer to [CenterNet-Better](https://github.com/FateScript/CenterNet-better), and make the following changes
  - fix wrong image mean and variance in image normalization to be compatible with the pre-trained backbone.
  - Use SGD rather than ADAM optimizer and add warmup and grad clip.
  - Use DistributedDataParallel as other models in MMDetection rather than using DataParallel.
