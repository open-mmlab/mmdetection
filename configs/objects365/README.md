# Objects365 Dataset

> [Objects365 Dataset](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shao_Objects365_A_Large-Scale_High-Quality_Dataset_for_Object_Detection_ICCV_2019_paper.pdf)

<!-- [DATASET] -->

## Abstract

<!-- [ABSTRACT] -->

#### Objects365 Dataset V1

[Objects365 Dataset V1](http://www.objects365.org/overview.html) is a brand new dataset,
designed to spur object detection research with a focus on diverse objects in the Wild.
It has 365 object categories over 600K training images. More than 10 million, high-quality bounding boxes are manually labeled through a three-step, carefully designed annotation pipeline. It is the largest object detection dataset (with full annotation) so far and establishes a more challenging benchmark for the community. Objects365 can serve as a better feature learning dataset for localization-sensitive tasks like object detection
and semantic segmentation.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/48282753/208368046-b7573022-06c9-4a99-af17-a6ac7407e3d8.png" height="400"/>
</div>

#### Objects365 Dataset V2

[Objects365 Dataset V2](http://www.objects365.org/overview.html) is based on the V1 release of the Objects365 dataset.
Objects 365 annotated 365 object classes on more than 1800k images, with more than 29 million bounding boxes in the training set, surpassing PASCAL VOC, ImageNet, and COCO datasets.
Objects 365 includes 11 categories of people, clothing, living room, bathroom, kitchen, office/medical, electrical appliances, transportation, food, animals, sports/musical instruments, and each category has dozens of subcategories.

## Citation

```
@inproceedings{shao2019objects365,
  title={Objects365: A large-scale, high-quality dataset for object detection},
  author={Shao, Shuai and Li, Zeming and Zhang, Tianyuan and Peng, Chao and Yu, Gang and Zhang, Xiangyu and Li, Jing and Sun, Jian},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={8430--8439},
  year={2019}
}
```

## Prepare Dataset

1. You need to download and extract Objects365 dataset. Users can download Objects365 V2 by using `tools/misc/download_dataset.py`.

   **Usage**

   ```shell
   python tools/misc/download_dataset.py --dataset-name objects365v2 \
   --save-dir ${SAVING PATH} \
   --unzip \
   --delete  # Optional, delete the download zip file
   ```

   **Note:** There is no download link for Objects365 V1 right now. If you would like to download Objects365-V1, please visit [official website](http://www.objects365.org/) to concat the author.

2. The directory should be like this:

   ```none
   mmdetection
   ├── mmdet
   ├── tools
   ├── configs
   ├── data
   │   ├── Objects365
   │   │   ├── Obj365_v1
   │   │   │   ├── annotations
   │   │   │   │   ├── objects365_train.json
   │   │   │   │   ├── objects365_val.json
   │   │   │   ├── train        # training images
   │   │   │   ├── val          # validation images
   │   │   ├── Obj365_v2
   │   │   │   ├── annotations
   │   │   │   │   ├── zhiyuan_objv2_train.json
   │   │   │   │   ├── zhiyuan_objv2_val.json
   │   │   │   ├── train        # training images
   │   │   │   │   ├── patch0
   │   │   │   │   ├── patch1
   │   │   │   │   ├── ...
   │   │   │   ├── val          # validation images
   │   │   │   │   ├── patch0
   │   │   │   │   ├── patch1
   │   │   │   │   ├── ...
   ```

## Results and Models

### Objects365 V1

| Architecture | Backbone |  Style  | Lr schd | Mem (GB) | box AP |                                                              Config                                                               |                                                                                                                                                                                Download                                                                                                                                                                                |
| :----------: | :------: | :-----: | :-----: | :------: | :----: | :-------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Faster R-CNN |   R-50   | pytorch |   1x    |    -     |  19.6  |   [config](https://github.com/open-mmlab/mmdetection/tree/main/configs/objects365/faster-rcnn_r50_fpn_16xb4-1x_objects365v1.py)   |           [model](https://download.openmmlab.com/mmdetection/v2.0/objects365/faster_rcnn_r50_fpn_16x4_1x_obj365v1/faster_rcnn_r50_fpn_16x4_1x_obj365v1_20221219_181226-9ff10f95.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/objects365/faster_rcnn_r50_fpn_16x4_1x_obj365v1/faster_rcnn_r50_fpn_16x4_1x_obj365v1_20221219_181226.log.json)           |
| Faster R-CNN |   R-50   | pytorch |  1350K  |    -     |  22.3  | [config](https://github.com/open-mmlab/mmdetection/tree/main/configs/objects365/faster-rcnn_r50-syncbn_fpn_1350k_objects365v1.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/objects365/faster_rcnn_r50_fpn_syncbn_1350k_obj365v1/faster_rcnn_r50_fpn_syncbn_1350k_obj365v1_20220510_142457-337d8965.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/objects365/faster_rcnn_r50_fpn_syncbn_1350k_obj365v1/faster_rcnn_r50_fpn_syncbn_1350k_obj365v1_20220510_142457.log.json) |
|  Retinanet   |   R-50   | pytorch |   1x    |    -     |  14.8  |       [config](https://github.com/open-mmlab/mmdetection/tree/main/configs/objects365/retinanet_r50_fpn_1x_objects365v1.py)       |                         [model](https://download.openmmlab.com/mmdetection/v2.0/objects365/retinanet_r50_fpn_1x_obj365v1/retinanet_r50_fpn_1x_obj365v1_20221219_181859-ba3e3dd5.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/objects365/retinanet_r50_fpn_1x_obj365v1/retinanet_r50_fpn_1x_obj365v1_20221219_181859.log.json)                         |
|  Retinanet   |   R-50   | pytorch |  1350K  |    -     |  18.0  |  [config](https://github.com/open-mmlab/mmdetection/tree/main/configs/objects365/retinanet_r50-syncbn_fpn_1350k_objects365v1.py)  |     [model](https://download.openmmlab.com/mmdetection/v2.0/objects365/retinanet_r50_fpn_syncbn_1350k_obj365v1/retinanet_r50_fpn_syncbn_1350k_obj365v1_20220513_111237-7517c576.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/objects365/retinanet_r50_fpn_syncbn_1350k_obj365v1/retinanet_r50_fpn_syncbn_1350k_obj365v1_20220513_111237.log.json)     |

### Objects365 V2

| Architecture | Backbone |  Style  | Lr schd | Mem (GB) | box AP |                                                            Config                                                             |                                                                                                                                                                      Download                                                                                                                                                                      |
| :----------: | :------: | :-----: | :-----: | :------: | :----: | :---------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Faster R-CNN |   R-50   | pytorch |   1x    |    -     |  19.8  | [config](https://github.com/open-mmlab/mmdetection/tree/main/configs/objects365/faster-rcnn_r50_fpn_16xb4-1x_objects365v2.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/objects365/faster_rcnn_r50_fpn_16x4_1x_obj365v2/faster_rcnn_r50_fpn_16x4_1x_obj365v2_20221220_175040-5910b015.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/objects365/faster_rcnn_r50_fpn_16x4_1x_obj365v2/faster_rcnn_r50_fpn_16x4_1x_obj365v2_20221220_175040.log.json) |
|  Retinanet   |   R-50   | pytorch |   1x    |    -     |  16.7  |     [config](https://github.com/open-mmlab/mmdetection/tree/main/configs/objects365/retinanet_r50_fpn_1x_objects365v2.py)     |               [model](https://download.openmmlab.com/mmdetection/v2.0/objects365/retinanet_r50_fpn_1x_obj365v2/retinanet_r50_fpn_1x_obj365v2_20221223_122105-d9b191f1.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/objects365/retinanet_r50_fpn_1x_obj365v2/retinanet_r50_fpn_1x_obj365v2_20221223_122105.log.json)               |
