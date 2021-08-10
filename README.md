


## CenterNet2 - MMDetection
This repository is the code that submitted for [OpenMMLab Algorithm Ecological Challenge](https://openmmlab.com/competitions/algorithm-2021)，the paper reproduced is [Probabilistic two-stage detection](https://arxiv.org/pdf/2103.07461.pdf)




### How To Use？
Our project is based on [mmdetecion](https://github.com/open-mmlab/mmdetection), please refer [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) for the basic usage of MMDetection. 

- **copy new files to mmdetection project**

```bash
cd CenterNet2-MMDetection
mv ./configs/centernet2 ${mmdetection}/configs/
mv ./configs/_base_/models/centernet2_cascade_r50_fpn.py ${mmdetection}/configs/_base_/models/
mv ./mmdet/models/detectors/centernet2.py ${mmdetection}/mmdet/models/detectors/
mv ./mmdet/models/dense_heads/custom_centernet_head.py ${mmdetection}/mmdet/models/dense_heads/
mv ./mmdet/models/roi_heads/custom_cascade_roi_head.py ${mmdetection}/mmdet/models/roi_heads/
```


- **register module to  `__init__.py`**

*mmdetection/models/detectors/__init__.py*


```python
...
from .centernet2 import CenterNet2
__all__ = [
    ..., 'CenterNet2'
]
```


*mmdetection/models/dense_heads/__init__.py*

```python
...
from .custom_centernet_head import CustomCenterNetHead
__all__ = [
    ..., 'CustomCenterNetHead'
]
```
*mmdetection/models/roi_heads/__init__.py*

```python
...
from .custom_cascade_roi_head import CustomCascadeRoIHead
__all__ = [
    ..., 'CustomCascadeRoIHead'
]
```

## Train and Test
- **train**

```bash
# single-gpu
cd ${mmdetection}/tools
python ./train.py  ../configs/centernet2/centernet2_cascade_res50_fpn_1x_coco.py [optional arguments]
# multi-gpu
./dist_train.sh ../configs/centernet2/centernet2_cascade_res50_fpn_1x_coco.py ${GPU_NUM} [optional arguments]
```

- **test**
```bash
# single-gpu
cd ${mmdetection}/tools
python ./test.py  ../configs/centernet2/centernet2_cascade_res50_fpn_1x_coco.py ${CHECKPOINT_FILE} [optional arguments]
# multi-gpu
./dist_test.sh ../configs/centernet2/centernet2_cascade_res50_fpn_1x_coco.py ${CHECKPOINT_FILE} ${GPU_NUM} --out ${RESULT_FILE} [optional arguments]
```




## Model zoo
pre-trained model in Baidunetdisk


 
| name | bbox_map |download|
|--|--|--|
| CenterNet2 | 40.4 |model |



## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

