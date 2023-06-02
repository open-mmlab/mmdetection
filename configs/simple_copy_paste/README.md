# SimpleCopyPaste

> [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://arxiv.org/abs/2012.07177)

<!-- [ALGORITHM] -->

## Abstract

Building instance segmentation models that are data-efficient and can handle rare object categories is an important challenge in computer vision. Leveraging data augmentations is a promising direction towards addressing this challenge. Here, we perform a systematic study of the Copy-Paste augmentation (\[13, 12\]) for instance segmentation where we randomly paste objects onto an image. Prior studies on Copy-Paste relied on modeling the surrounding visual context for pasting the objects. However, we find that the simple mechanism of pasting objects randomly is good enough and can provide solid gains on top of strong baselines. Furthermore, we show Copy-Paste is additive with semi-supervised methods that leverage extra data through pseudo labeling (e.g. self-training). On COCO instance segmentation, we achieve 49.1 mask AP and 57.3 box AP, an improvement of +0.6 mask AP and +1.5 box AP over the previous state-of-the-art. We further demonstrate that Copy-Paste can lead to significant improvements on the LVIS benchmark. Our baseline model outperforms the LVIS 2020 Challenge winning entry by +3.6 mask AP on rare categories.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/161843866-c5b769da-58b2-4c1f-8078-db4a4ded3881.png"/>
</div>

## Results and Models

### Mask R-CNN with Standard Scale Jittering (SSJ) and Simple Copy-Paste(SCP)

Standard Scale Jittering(SSJ) resizes and crops an image with a resize range of 0.8 to 1.25 of the original image size, and Simple Copy-Paste(SCP) selects a random subset of objects from one of the images and pastes them onto the other image.

| Backbone | Training schedule | Augmentation | batch size | box AP | mask AP |                                                                           Config                                                                           |                                                                                                                                                                                                                               Download                                                                                                                                                                                                                               |
| :------: | :---------------: | :----------: | :--------: | :----: | :-----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   |        90k        |     SSJ      |     64     |  43.3  |  39.0   |   [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_90k_coco.py)    |           [model](https://download.openmmlab.com/mmdetection/v2.0/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_90k_coco/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_90k_coco_20220316_181409-f79c84c5.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_90k_coco/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_90k_coco_20220316_181409.log.json)           |
|   R-50   |        90k        |   SSJ+SCP    |     64     |  43.8  |  39.2   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_90k_coco.py)  |   [model](https://download.openmmlab.com/mmdetection/v2.0/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_90k_coco/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_90k_coco_20220316_181307-6bc5726f.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_90k_coco/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_90k_coco_20220316_181307.log.json)   |
|   R-50   |       270k        |     SSJ      |     64     |  43.5  |  39.1   |   [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_270k_coco.py)   |         [model](https://download.openmmlab.com/mmdetection/v2.0/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_270k_coco/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_270k_coco_20220324_182940-33a100c5.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_270k_coco/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_270k_coco_20220324_182940.log.json)         |
|   R-50   |       270k        |   SSJ+SCP    |     64     |  45.1  |  40.3   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco_20220324_201229-80ee90b7.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco_20220324_201229.log.json) |

## Citation

```latex
@inproceedings{ghiasi2021simple,
  title={Simple copy-paste is a strong data augmentation method for instance segmentation},
  author={Ghiasi, Golnaz and Cui, Yin and Srinivas, Aravind and Qian, Rui and Lin, Tsung-Yi and Cubuk, Ekin D and Le, Quoc V and Zoph, Barret},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2918--2928},
  year={2021}
}
```
