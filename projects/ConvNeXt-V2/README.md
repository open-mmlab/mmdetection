# ConvNeXt-V2

> [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](http://arxiv.org/abs/2301.00808)

## Abstract

Driven by improved architectures and better representation learning frameworks, the field of visual recognition has enjoyed rapid modernization and performance boost in the early 2020s. For example, modern ConvNets, represented by ConvNeXt \[52\], have demonstrated strong performance in various scenarios. While these models were originally designed for supervised learning with ImageNet labels, they can also potentially benefit from self-supervised learning techniques such as masked autoencoders (MAE) . However, we found that simply combining these two approaches leads to subpar performance. In this paper, we propose a fully convolutional masked autoencoder framework and a new Global Response Normalization (GRN) layer that can be added to the ConvNeXt architecture to enhance inter-channel feature competition. This co-design of self-supervised learning techniques and architectural improvement results in a new model family called ConvNeXt V2, which significantly improves the performance of pure ConvNets on various recognition benchmarks, including ImageNet classification, COCO detection, and ADE20K segmentation. We also provide pre-trained ConvNeXt V2 models of various sizes, ranging from an efficient 3.7Mparameter Atto model with 76.7% top-1 accuracy on Im-ageNet, to a 650M Huge model that achieves a state-of-theart 88.9% accuracy using only public training data.

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/212588579-02d621d8-5796-4f0d-b4d2-758fe9c2f395.png" width="50%"/>
</div>

## Results and models

|   Method   |   Backbone    | Pretrain | Lr schd | Augmentation | Mem (GB) | box AP | mask AP |                            Config                            |                                                                                                                                                                                        Download                                                                                                                                                                                         |
| :--------: | :-----------: | :------: | :-----: | :----------: | :------: | :----: | :-----: | :----------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Mask R-CNN | ConvNeXt-V2-B |  FCMAE   |   3x    |     LSJ      |   22.5   |  52.9  |  46.4   | [config](./mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/convnextv2/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco_20230113_110947-757ee2dd.pth)  \| [log](https://download.openmmlab.com/mmdetection/v3.0/convnextv2/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco_20230113_110947.log.json) |

**Note**:

- This is a pre-release version of ConvNeXt-V2 object detection. The official finetuning setting of ConvNeXt-V2 has not been released yet.
- ConvNeXt backbone needs to install [MMPretrain](https://github.com/open-mmlab/mmpretrain/) first, which has abundant backbones for downstream tasks.

```shell
pip install mmpretrain
```

## Citation

```bibtex
@article{Woo2023ConvNeXtV2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon and Saining Xie},
  year={2023},
  journal={arXiv preprint arXiv:2301.00808},
}
```
