# InstaBoost for MMDetection

Configs in this directory is the implementation for ICCV2019 paper "InstaBoost: Boosting Instance Segmentation Via Probability Map Guided Copy-Pasting" and provided by the authors of the paper. InstaBoost is a data augmentation method for object detection and instance segmentation. The paper has been released on [`arXiv`](https://arxiv.org/abs/1908.07801).

```
@inproceedings{fang2019instaboost,
  title={Instaboost: Boosting instance segmentation via probability map guided copy-pasting},
  author={Fang, Hao-Shu and Sun, Jianhua and Wang, Runzhong and Gou, Minghao and Li, Yong-Lu and Lu, Cewu},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={682--691},
  year={2019}
}
```

## Usage

### Requirements

You need to install `instaboostfast` before using it.

```
pip install instaboostfast
```

The code and more details can be found [here](https://github.com/GothicAi/Instaboost).

### Integration with MMDetection

InstaBoost have been already integrated in the data pipeline, thus all you need is to add or change **InstaBoost** configurations after **LoadImageFromFile**. We have provided examples like [this](mask_rcnn_r50_fpn_instaboost_4x.py#L121). You can refer to [`InstaBoostConfig`](https://github.com/GothicAi/InstaBoost-pypi#instaboostconfig) for more details.

## Results and Models

 - All models were trained on `coco_2017_train` and tested on `coco_2017_val` for conveinience of evaluation and comparison. In the paper, the results are obtained from `test-dev`.
 - To balance accuracy and training time when using InstaBoost, models released in this page are all trained for 48 Epochs. Other training and testing configs strictly follow the original framework.
 - The results and models are provided by the [authors](https://github.com/GothicAi/Instaboost) (many thanks).


|    InstaBoost   |     Network     |       Backbone       | Lr schd |      box AP       |      mask AP       |      Download       |
| :-------------: | :-------------: |      :--------:      | :-----: |      :----:       |      :-----:       | :-----------------: |
|   ×   |    Mask R-CNN   |       R-50-FPN       |   1x    |  37.3  |  34.2   | - |
|   √   |    Mask R-CNN   |       R-50-FPN       |   4x    |**40.0**|**36.2** |[Baidu](https://pan.baidu.com/s/1PLn1K5qreDoM4wh7nbsLqA) / [Google](https://drive.google.com/file/d/1uUT1qc3oYS8xHLyM7bJWgxBNbW-9sa1f/view?usp=sharing)|
|   ×   |    Mask R-CNN   |      R-101-FPN       |   1x    |  39.4  |  35.9   | - |
|   √   |    Mask R-CNN   |      R-101-FPN       |   4x    |**42.1**|**37.8** |[Baidu](https://pan.baidu.com/s/1IZpqCDrcrOiwNJ-Y_3wpOQ) / [Google](https://drive.google.com/file/d/1idGMPexovIDUHXSNlpIA1mjKzgnFrcW3/view?usp=sharing)|
|   ×   |    Mask R-CNN   |   X-101-64x4d-FPN    |   1x    |  42.1  |  38.0   | - |
|   ×   |    Mask R-CNN   |   X-101-64x4d-FPN    |   2x    |  *42.0*  |  *37.7*   | - |
|   √   |    Mask R-CNN   |   X-101-64x4d-FPN    |   4x    |**44.5**|**39.5** |[Baidu](https://pan.baidu.com/s/1KrHQBHcHjWONpXbC2qUzxw) / [Google](https://drive.google.com/file/d/1qD4V9uYbtpaZBmTMTgP7f0uw46zroY9-/view?usp=sharing)|
|   ×   |  Cascade R-CNN  |       R-101-FPN      |   1x    |  42.6  |  37.0   | - |
|   √   |  Cascade R-CNN  |       R-101-FPN      |   4x    |**45.4**|**39.2** |[Baidu](https://pan.baidu.com/s/1_4cJ0B9fugcA-oBHYe9o_A) / [Google](https://drive.google.com/file/d/1xhiuFoOMQyDIvOrz6MiAZPboRRe1YK8p/view?usp=sharing)|
|   ×   |  Cascade R-CNN  |   X-101-64x4d-FPN    |   1x    |  45.4  |  39.1   | - |
|   √   |  Cascade R-CNN  |   X-101-64x4d-FPN    |   4x    |**47.2**|**40.4** |[Baidu](https://pan.baidu.com/s/1nu73IpRbTEb4caPMHWJMXA) / [Google](https://drive.google.com/file/d/11iaKH-ZeVCi-65wzlT5OxxUOkREMzXRW/view?usp=sharing)|
|   ×   |       SSD       |      VGG16-512       |   120e  |  29.3  |   -     | - |
|   √   |       SSD       |      VGG16-512       |   360e  |**30.3**|   -     |[Baidu](https://pan.baidu.com/s/1G-1atZ81A8mLLx8taJAuwQ) / [Google](https://drive.google.com/file/d/1sqMIEusZw2Y7Ge8DuJgmhSP-2V74BNKy/view?usp=sharing)|
