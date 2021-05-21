# CenterNet

## Introduction

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
| ResNet-18 | N | 3.45 | 26.0 | 27.4 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/centernet/centernet_resnet18_140e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210519_092334-eafe8ccd.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210519_092334.log.json) |
| ResNet-18 | Y | 3.47 | 29.5 | 31.0 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/centernet/centernet_resnet18_dcnv2_140e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco_20210520_101209-da388ba2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco_20210520_101209.log.json) |

Note:

- Flip box AP setting is single-scale and `flip=True`.
- Due to complex data enhancement, we find that the performance is unstable and may fluctuate by about 0.4 mAP. mAP 29.4 ~ 29.8 is acceptable in ResNet-18-DCNv2.
- Compared to the source code, we refer to [CenterNet-Better](https://github.com/FateScript/CenterNet-better), and make the following changes
  - fix wrong image mean and variance in image normalization to be compatible with the pre-trained backbone.
  - Use SGD rather than ADAM optimizer and add warmup and grad clip.
  - Use DistributedDataParallel as other models in MMDetection rather than using DataParallel.
