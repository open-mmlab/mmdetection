# CARAFE: Content-Aware ReAssembly of FEatures

## Introduction

We provide config files to reproduce the object detection & instance segmentation results in the ICCV 2019 Oral paper for [CARAFE: Content-Aware ReAssembly of FEatures](https://arxiv.org/abs/1905.02188).

```
@inproceedings{Wang_2019_ICCV,
    title = {CARAFE: Content-Aware ReAssembly of FEatures},
    author = {Wang, Jiaqi and Chen, Kai and Xu, Rui and Liu, Ziwei and Loy, Chen Change and Lin, Dahua},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

## Results and Models

The results on COCO 2017 val is shown in the below table.

| Method                 | Backbone | Style   | Lr schd | Test Proposal Num| Box AP | Mask AP | Download |
| :--------------------: | :------: | :-----: |:-------:| :--------------: | :----: | :--------: |:-------: |
| Faster R-CNN w/ CARAFE | R-50-FPN | pytorch | 1x      | 1000 |           |        |  |
| -                      |    -     |  -      | -       | 2000 |           |        |  |
| Mask R-CNN w/ CARAFE   | R-50-FPN | pytorch | 1x      | 1000 |           |        |  |
| -                      |   -      |  -      |   -     | 2000 |           |        |  |

## Implementation

The CUDA implementation of CARAFE can be find at `mmdet/ops/carafe` under this repository.

## Setup CARAFE

a. Use CARAFE in mmdetection.

Install mmdetection following the official guide.

b. Use CARAFE in your own project.

Git clone mmdetection.
```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
```
Setup CARAFE in your own project.
```shell
cp -r ./mmdet/ops/carafe $Your_Project_Path$
cd $Your_Project_Path$/carafe
python setup.py develop
# or "pip install -v -e ."
cd ..
python ./carafe/grad_check.py
```
