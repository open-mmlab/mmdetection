# <img src="v3det_icon.jpg" height="40"> V3Det: Vast Vocabulary Visual Detection Dataset (ICCV 2023)

> V3Det: Vast Vocabulary Visual Detection Dataset (ICCV 2023) [[Paper](https://arxiv.org/pdf/2304.03752.pdf), [Dataset](https://v3det.openxlab.org.cn/)]   
> [Jiaqi Wang](https://myownskyw7.github.io/), [Pan Zhang](https://panzhang0212.github.io/), Tao Chu, Yuhang Cao, Yujie Zhou, [Tong Wu](https://wutong16.github.io/), Bin Wang, Conghui He, [Dahua Lin](http://dahua.site/)    

<p align="left">
    <img width=960 src="introduction.jpg"/>
</p>


<!-- [ALGORITHM] -->

## Abstract

Recent advances in detecting arbitrary objects in the real world are trained and evaluated on object detection datasets with a relatively restricted vocabulary. To facilitate the development of more general visual object detection, we propose V3Det, a vast vocabulary visual detection dataset with precisely annotated bounding boxes on massive images. V3Det has several appealing properties: 1) Vast Vocabulary: It contains bounding boxes of objects from 13,029 categories on real-world images, which is 10 times larger than the existing large vocabulary object detection dataset, e.g., LVIS. 2) Hierarchical Category Organization: The vast vocabulary of V3Det is organized by a hierarchical category tree which annotates the inclusion relationship among categories, encouraging the exploration of category relationships in vast and open vocabulary object detection. 3) Rich Annotations: V3Det comprises precisely annotated objects in 245k images and professional descriptions of each category written by human experts and a powerful chatbot. By offering a vast exploration space, V3Det enables extensive benchmarks on both vast and open vocabulary object detection, leading to new observations, practices, and insights for future research. It has the potential to serve as a cornerstone dataset for developing more general visual perception systems.

## Prepare Dataset

Please download and prepare V3Det Dataset at [V3Det Homepage](https://v3det.openxlab.org.cn/) and [V3Det Github](https://github.com/V3Det/V3Det).

The data includes a training set, a validation set, comprising 13,204 categories. The training set consists of 183,354 images, while the validation set has 29,821 images. The data organization is:
```
data/
    images/
        <category_node>/
            |────<image_name>.png
            ...
        ...
    annotations/
        |────v3det_2023_v1_category_tree.json       # Category tree
        |────category_name_13204_v3det_2023_v1.txt  # Category name
        |────v3det_2023_v1_train.json               # Train set
        |────v3det_2023_v1_val.json                 # Validation set
```


## Results and Models


| Backbone |      Model      | Lr schd | box AP |                     Config                      |                                                                                                                                                     Download                                                                                                                                                      |
|:--------:|:---------------:|:-------:|:------:| :---------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   |  Faster R-CNN   |   2x    |  25.4  |  [config](./faster_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x.py)   |    [model](https://download.openxlab.org.cn/models/V3Det/V3Det/weight//faster_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x)
|   R-50   |  Cascade R-CNN  |   2x    |  31.6  |  [config](./cascade_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x.py)   |    [model](https://download.openxlab.org.cn/models/V3Det/V3Det/weight//cascade_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x) 
|   R-50   |      FCOS       |   2x    |  9.4   |  [config](./fcos_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x.py)   |    [model](https://download.openxlab.org.cn/models/V3Det/V3Det/weight//fcos_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x)
|   R-50   | Deformable-DETR |   50e   |  34.4  |  [config](./deformable-detr-refine-twostage_r50_8xb4_sample1e-3_v3det_50e.py)   |    [model](https://download.openxlab.org.cn/models/V3Det/V3Det/weight/Deformable_DETR_V3Det_R50)
|   R-50   |      DINO       |   36e   |  33.5  |  [config](./dino-4scale_r50_8xb2_sample1e-3_v3det_36e.py)   |    [model](https://download.openxlab.org.cn/models/V3Det/V3Det/weight/DINO_V3Det_R50)
|  Swin-B  |  Faster R-CNN   |   2x    |  37.6  |  [config](./faster_rcnn_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x.py)   |    [model](https://download.openxlab.org.cn/models/V3Det/V3Det/weight//faster_rcnn_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x)
|  Swin-B  |  Cascade R-CNN  |   2x    |  42.5  |  [config](./cascade_rcnn_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x.py)   |    [model](https://download.openxlab.org.cn/models/V3Det/V3Det/weight//cascade_rcnn_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x) 
|  Swin-B  |      FCOS       |   2x    |  21.0  |  [config](./fcos_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x.py)   |    [model](https://download.openxlab.org.cn/models/V3Det/V3Det/weight//fcos_swinb_fpn_8x4_sample1e-3_mstrain_v3det_2x)
|  Swin-B  | Deformable-DETR |   50e   |  42.5  |  [config](./deformable-detr-refine-twostage_swin_16xb2_sample1e-3_v3det_50e.py)   |    [model](https://download.openxlab.org.cn/models/V3Det/V3Det/weight/Deformable_DETR_V3Det_SwinB)
|  Swin-B  |      DINO       |   36e   |  42.0  |  [config](./dino-4scale_swin_16xb1_sample1e-3_v3det_36e.py)   |    [model](https://download.openxlab.org.cn/models/V3Det/V3Det/weight/DINO_V3Det_SwinB)



## Citation

```latex
@inproceedings{wang2023v3det,
      title = {V3Det: Vast Vocabulary Visual Detection Dataset}, 
      author = {Wang, Jiaqi and Zhang, Pan and Chu, Tao and Cao, Yuhang and Zhou, Yujie and Wu, Tong and Wang, Bin and He, Conghui and Lin, Dahua},
      booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
      month = {October},
      year = {2023}
}
```
