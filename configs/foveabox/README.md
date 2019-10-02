# FoveaBox: Beyond Anchor-based Object Detector

FoveaBox is an accurate, flexible and completely anchor-free object detection system for object detection framework, as presented in our paper [https://arxiv.org/abs/1904.03797](https://arxiv.org/abs/1904.03797):
Different from previous anchor-based methods, FoveaBox directly learns the object existing possibility and the bounding box coordinates without anchor reference. This is achieved by: (a) predicting category-sensitive semantic maps for the object existing possibility, and (b) producing category-agnostic bounding box for each position that potentially contains an object.

## Main Results
### Results on R50/101-FPN 

| Backbone  | Style   |  align  | ms-train| Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50      | pytorch | N       | N       | 1x      | 5.7      | -                   |                | 36.5   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/foveabox/fovea_r50_fpn_4gpu_1x_20190905-3b185a5d.pth) |
| R-50      | pytorch | N       | N       | 2x      | -        | -                   |                | 36.9   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/foveabox/fovea_r50_fpn_4gpu_2x_20190905-4a07f6e0.pth) |
| R-50      | pytorch | Y       | N       | 2x      | -        | -                   |                | 37.9   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/foveabox/fovea_align_gn_r50_fpn_4gpu_2x_20190905-3e6bc82f.pth) |
| R-50      | pytorch | Y       | Y       | 2x      | -        | -                   |                | 40.1   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/foveabox/fovea_align_gn_ms_r50_fpn_4gpu_2x_20190905-13374f33.pth) |
| R-101     | pytorch | N       | N       | 1x      | 9.4      | -                   |                | 38.5   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/foveabox/fovea_r101_fpn_4gpu_1x_20190905-80ff93a6.pth) |
| R-101     | pytorch | N       | N       | 2x      | -        | -                   |                | 38.5   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/foveabox/fovea_r101_fpn_4gpu_2x_20190905-d9c99fb1.pth) |
| R-101     | pytorch | Y       | N       | 2x      | -        | -                   |                | 39.4   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/foveabox/fovea_align_gn_r101_fpn_4gpu_2x_20190905-407ddad6.pth) |
| R-101     | pytorch | Y       | Y       | 2x      | -        | -                   |                | 41.9   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/foveabox/fovea_align_gn_ms_r101_fpn_4gpu_2x_20190905-936c7277.pth) |

[1] *1x and 2x mean the model is trained for 12 and 24 epochs, respectively.* \
[2] *Align means utilizing deformable convolution to align the cls branch.* \
[3] *All results are obtained with a single model and without any test time data augmentation.*\
[4] *We use 4 NVIDIA Tesla V100 GPUs for training.*

Any pull requests or issues are welcome.

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@article{kong2019foveabox,
  title={FoveaBox: Beyond Anchor-based Object Detector},
  author={Kong, Tao and Sun, Fuchun and Liu, Huaping and Jiang, Yuning and Shi, Jianbo},
  journal={arXiv preprint arXiv:1904.03797},
  year={2019}
}
```