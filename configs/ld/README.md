# Localization Distillation for Object Detection

## Abstract

<!-- [ABSTRACT] -->

Knowledge distillation (KD) has witnessed its powerful capability in learning compact models in object detection. Previous KD methods for object detection mostly focus on imitating deep features within the imitation regions instead of mimicking classification logits due to its inefficiency in distilling localization information. In this paper, by reformulating the knowledge distillation process on localization, we present a novel localization distillation (LD) method which can efficiently transfer the localization knowledge from the teacher to the student. Moreover, we also heuristically introduce the concept of valuable localization region that can aid to selectively distill the semantic and localization knowledge for a certain region. Combining these two new components, for the first time, we show that logit mimicking can outperform feature imitation and localization knowledge distillation is more important and efficient than semantic knowledge for distilling object detectors. Our distillation scheme is simple as well as effective and can be easily applied to different dense object detectors. Experiments show that our LD can boost the AP score of GFocal-ResNet-50 with a single-scale 1× training schedule from 40.1 to 42.1 on the COCO benchmark without any sacrifice on the inference speed.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143966265-48a03668-8585-4525-8a86-afa2209d1602.png"/>
</div>

<!-- [PAPER_TITLE: Localization Distillation for Object Detection] -->
<!-- [PAPER_URL: https://arxiv.org/abs/2102.12252] -->

## Citation

<!-- [ALGORITHM] -->

```latex
@Article{zheng2021LD,
  title={Localization Distillation for Object Detection},
  author= {Zhaohui Zheng, Rongguang Ye, Ping Wang, Jun Wang, Dongwei Ren, Wangmeng Zuo},
  journal={arXiv:2102.12252},
  year={2021}
}
```

### GFocalV1 with LD

|  Teacher  | Student | Training schedule | Mini-batch size | AP (val) | AP50 (val) | AP75 (val) | Config |
| :-------: | :-----: | :---------------: | :-------------: | :------: | :--------: | :--------: |  :--------------: |
|    --     |  R-18   |        1x         |        6        |   35.8   |    53.1    |    38.2    |          |
|   R-101   |  R-18   |        1x         |        6        |   36.5   |    52.9    |    39.3    |   [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ld/ld_r18_gflv1_r101_fpn_coco_1x.py)          |
|    --     |  R-34   |        1x         |        6        |   38.9   |    56.6    |    42.2    |          |
|   R-101   |  R-34   |        1x         |        6        |   39.8   |    56.6    |    43.1    |     [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ld/ld_r34_gflv1_r101_fpn_coco_1x.py)        |
|    --     |  R-50   |        1x         |        6        |   40.1   |    58.2    |    43.1    |            |
|   R-101   |  R-50   |        1x         |        6        |   41.1   |    58.7    |    44.9    |    [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ld/ld_r50_gflv1_r101_fpn_coco_1x.py)        |
|    --     |  R-101  |        2x         |        6        |   44.6   |    62.9    |    48.4    |           |
| R-101-DCN |  R-101  |        2x         |        6        |   45.4   |    63.1    |    49.5    | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ld/ld_r101_gflv1_r101dcn_fpn_coco_1x.py)           |

## Note

- Meaning of Config name: ld_r18(student model)_gflv1(based on gflv1)_r101(teacher model)_fpn(neck)_coco(dataset)_1x(12 epoch).py
