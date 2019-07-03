# Panoptic FPN

## Introduction

```
@inproceedings{Kirillov_2019_CVPR,
  title={Panoptic Feature Pyramid Networks},
  author={Kirillov, Alexander and Girshick, Ross and He, Kaiming and Dollar, Piotr},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Preparation

a. Download 2017 [Panopatic Train/Val annotations](http://cocodataset.org/#download) for COCO 2017

b. Install [COCO 2018 Panoptic Segmentation Task API](https://github.com/cocodataset/panopticapi)

c. symlink the panoptic segmentation annotations root to `$MMDETECTION/data/coco/PanopticSegm_annotations`

  Note: The path of panoptic segmentaion annotations should be `$MMDETECTION/data/coco/PanopticSegm_annotations/PanopticSegm_annotations/`

d. Extract semantic segmentation from data in COCO panoptic format by using the script `$PANOPTICAPI/converters/panoptic2semantic_segmentation.py`

  Note: The path of semantic segmentaion annotations should be `$MMDETECTION/data/coco/PanopticSegm_annotations/SemanticSegm_annotations/semantic_val2017` and `$MMDETECTION/data/coco/PanopticSegm_annotations/SemanticSegm_annotations/semantic_train2017`

## Evaluation

a. using the script `$PANOPTICAPI/combine_semantic_and_instance_predictions.py` to get panoptic segmentation results

b. using the script `$PANOPTICAPI/evaluation.py` to evaluate panoptic segmentation results

## Results and Models

| Backbone    | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | PQ
|:-----------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|
| R-50        | 1x      | 9.6      | 0.831               |5.8            | 38.8   | 

The model is trained and tested on 8 Titan Xp GPUs