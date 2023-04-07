# iSAID数据集

> **iSAID**: A Large-scale Dataset for Instance Segmentation in Aerial Images

## 数据集介绍

Existing Earth Vision datasets are either suitable for semantic segmentation or object detection. iSAID is the first benchmark dataset for instance segmentation in aerial images. This large-scale and densely annotated dataset contains 655,451 object instances for 15 categories across 2,806 high-resolution images. The distinctive characteristics of iSAID are the following: (a) large number of images with high spatial resolution, (b) fifteen important and commonly occurring categories, (c) large number of instances per category, (d) large count of labelled instances per image, which might help in learning contextual information, (e) huge object scale variation, containing small, medium and large objects, often within the same image, (f) Imbalanced and uneven distribution of objects with varying orientation within images, depicting real-life aerial conditions, (g) several small size objects, with ambiguous appearance, can only be resolved with contextual reasoning, (h) precise instance-level annotations carried out by professional annotators, cross-checked and validated by expert annotators complying with well-defined guidelines.

For more detail, please refer to our [paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Zamir_iSAID_A_Large-scale_Dataset_for_Instance_Segmentation_in_Aerial_Images_CVPRW_2019_paper.pdf) .

## 数据集准备

iSAID数据集下载链接：[图像数据](https://captain-whu.github.io/DOTA/dataset.html)、[标注数据](https://captain-whu.github.io/iSAID/dataset.html)
请按照[官方仓库](https://github.com/CAPTAIN-WHU/iSAID_Devkit)中所述步骤进行数据预处理（`patch_width`=800,`patch_height`=800,`overlap_area`=200），最终得到的文件夹格式为

```
iSAID_patches
├── test
│   └── images
│       ├── P0006_0_0_800_800.png
│       └── ...
│       └── P0009_0_0_800_800.png
├── train
│   └── instance_only_filtered_train.json
│   └── images
│       ├── P0002_0_0_800_800_instance_color_RGB.png
│       ├── P0002_0_0_800_800_instance_id_RGB.png
│       ├── P0002_0_800_800.png
│       ├── ...
│       ├── P0010_0_0_800_800_instance_color_RGB.png
│       ├── P0010_0_0_800_800_instance_id_RGB.png
│       └── P0010_0_800_800.png
└── val
    └── instance_only_filtered_val.json
    └── images
        ├── P0003_0_0_800_800_instance_color_RGB.png
        ├── P0003_0_0_800_800_instance_id_RGB.png
        ├── P0003_0_0_800_800.png
        ├── ...
        ├── P0004_0_0_800_800_instance_color_RGB.png
        ├── P0004_0_0_800_800_instance_id_RGB.png
        └── P0004_0_0_800_800.png
```

之后，在mmdetection目录下使用以下命令转换json文件格式

```
python projects/iSAID/isaid_json.py /path/to/iSAID
```

## 使用方法

### 训练

```python
python tools/train.py projects/iSAID/configs/mask_rcnn_r50_fpn_1x_isaid.py
```

### 测试

```python
python tools/test.py projects/iSAID/configs/mask_rcnn_r50_fpn_1x_isaid.py ${CHECKPOINT_PATH}
```

## Citation

```
@inproceedings{waqas2019isaid,
title={iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images},
author={Waqas Zamir, Syed and Arora, Aditya and Gupta, Akshita and Khan, Salman and Sun, Guolei and Shahbaz Khan, Fahad and Zhu, Fan and Shao, Ling and Xia, Gui-Song and Bai, Xiang},
booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
pages={28--37},
year={2019}
}
```

```
@InProceedings{Xia_2018_CVPR,
author = {Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
title = {DOTA: A Large-Scale Dataset for Object Detection in Aerial Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
