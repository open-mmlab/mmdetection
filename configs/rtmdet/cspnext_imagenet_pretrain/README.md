# CSPNeXt ImageNet Pre-training

In this folder, we provide the imagenet pre-training config of RTMDet's backbone CSPNeXt.
To train with these configs, please install [MMClassification](https://github.com/open-mmlab/mmclassification) first.

## Results and Models

|    Model     | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                                             Download                                                             |
| :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :------------------------------------------------------------------------------------------------------------------------------: |
| CSPNeXt-tiny |  224x224   |   2.73    |  0.339   |   69.44   |   89.45   | [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext_tiny/cspnext-tiny_imagenet_600e.pth) |
|  CSPNeXt-s   |  224x224   |   4.89    |  0.664   |   74.41   |   92.23   |  [model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext_tiny/cspnext-s_imagenet_600e.pth)   |
