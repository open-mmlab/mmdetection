# Improving Object Detection by Label Assignment Distillation

<!-- [ALGORITHM] -->

```latex
@inproceedings{nguyen2021improving,
  title={Improving Object Detection by Label Assignment Distillation},
  author={Chuong H. Nguyen and Thuy C. Nguyen and Tuan N. Tang and Nam L. H. Phan},
  booktitle = {WACV},
  year={2022}
}
```

## Results and Models

We provide config files to reproduce the object detection results in the
WACV 2022 paper for Improving Object Detection by Label Assignment
Distillation.

### PAA with LAD

|  Teacher  | Student | Training schedule | AP (val) |                         Config                          |
| :-------: | :-----: | :---------------: | :------: |  :----------------------------------------------------: |
|    --     |  R-50   |        1x         |   40.4   |                                                         |
|    --     |  R-101  |        1x         |   42.6   |                                                         |
|   R-101   |  R-50   |        1x         |   41.6   |  [config](configs/lad/lad_r50_paa_r101_fpn_coco_1x.py)  |
|   R-50    |  R-101  |        1x         |   43.2   |  [config](configs/lad/lad_r101_paa_r50_fpn_coco_1x.py)  |

## Note

- Meaning of Config name: lad_r50(student model)_paa(based on paa)_r101(teacher model)_fpn(neck)_coco(dataset)_1x(12 epoch).py
- Results may fluctuate by about 0.2 mAP.
