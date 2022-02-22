# Open Images Dataset
<!-- [DATASET] -->

## Abstract

<!-- [ABSTRACT] -->
#### Open Images v6

[Open Images](https://storage.googleapis.com/openimages/web/index.html) is a dataset of ~9M images annotated with image-level labels,
object bounding boxes, object segmentation masks, visual relationships,
and localized narratives:

- It contains a total of 16M bounding boxes for 600 object classes on
1.9M images, making it the largest existing dataset with object location
annotations. The boxes have been largely manually drawn by professional
annotators to ensure accuracy and consistency. The images are very diverse
and often contain complex scenes with several objects (8.3 per image on
average).

- Open Images also offers visual relationship annotations, indicating pairs
of objects in particular relations (e.g. "woman playing guitar", "beer on
table"), object properties (e.g. "table is wooden"), and human actions (e.g.
"woman is jumping"). In total it has 3.3M annotations from 1,466 distinct
relationship triplets.

- In V5 we added segmentation masks for 2.8M object instances in 350 classes.
Segmentation masks mark the outline of objects, which characterizes their
spatial extent to a much higher level of detail.

- In V6 we added 675k localized narratives: multimodal descriptions of images
consisting of synchronized voice, text, and mouse traces over the objects being
described. (Note we originally launched localized narratives only on train in V6,
but since July 2020 we also have validation and test covered.)

- Finally, the dataset is annotated with 59.9M image-level labels spanning 19,957
classes.

We believe that having a single dataset with unified annotations for image
classification, object detection, visual relationship detection, instance
segmentation, and multimodal image descriptions will enable to study these
tasks jointly and stimulate progress towards genuine scene understanding.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/48282753/147199750-23e17230-c0cf-49a0-a13c-0d014d49107e.png" height="400"/>
</div>

#### Open Images Challenge 2019

[Open Images Challenges 2019](https://storage.googleapis.com/openimages/web/challenge2019.html) is based on the V5 release of the Open
Images dataset. The images of the dataset are very varied and
often contain complex scenes with several objects (explore the dataset).

## Citation

```
@article{OpenImages,
  author = {Alina Kuznetsova and Hassan Rom and Neil Alldrin and Jasper Uijlings and Ivan Krasin and Jordi Pont-Tuset and Shahab Kamali and Stefan Popov and Matteo Malloci and Alexander Kolesnikov and Tom Duerig and Vittorio Ferrari},
  title = {The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale},
  year = {2020},
  journal = {IJCV}
}
```

## Prepare Dataset

1. You need to download and extract Open Images dataset.

2. The Open Images dataset does not have image metas (width and height of the image),
which will be used during evaluation. We suggest to get test image metas before
training/testing by using `tools/misc/get_image_metas.py`.

    **Usage**
    ```shell
    python tools/misc/get_image_metas.py ${CONFIG} \
    --out ${OUTPUT FILE NAME}
    ```

3. The directory should be like this:

    ```none
    mmdetection
    ├── mmdet
    ├── tools
    ├── configs
    ├── data
    │   ├── OpenImages
    │   │   ├── annotations
    │   │   │   ├── bbox_labels_600_hierarchy.json
    │   │   │   ├── class-descriptions-boxable.csv
    │   │   │   ├── oidv6-train-annotations-bbox.scv
    │   │   │   ├── validation-annotations-bbox.csv
    │   │   │   ├── validation-annotations-human-imagelabels-boxable.csv    # is not necessary
    │   │   │   ├── validation-image-metas.pkl      # get from script
    │   │   ├── challenge2019
    │   │   │   ├── challenge-2019-train-detection-bbox.txt
    │   │   │   ├── challenge-2019-validation-detection-bbox.txt
    │   │   │   ├── class_label_tree.np
    │   │   │   ├── class_sample_train.pkl
    │   │   │   ├── challenge-2019-validation-detection-human-imagelabels.csv       # download from official website, not necessary
    │   │   │   ├── challenge-2019-validation-metas.pkl     # get from script
    │   │   ├── OpenImages
    │   │   │   ├── train           # training images
    │   │   │   ├── test            # testing images
    │   │   │   ├── validation      # validation images
    ```

**Note**:
1. The training and validation images of Open Images Challenge dataset are based on
Open Images v6, but the test images are different.
2. The Open Images Challenges annotations are obtained from [TSD](https://github.com/Sense-X/TSD).
You can also download the annotations from [official website](https://storage.googleapis.com/openimages/web/challenge2019_downloads.html),
and set data.train.type=OpenImagesDataset, data.val.type=OpenImagesDataset, and data.test.type=OpenImagesDataset in the config
3. If users do not want to use `validation-annotations-human-imagelabels-boxable.csv` and `challenge-2019-validation-detection-human-imagelabels.csv`
users can should set `data.val.load_image_level_labels=False` and `data.test.load_image_level_labels=False` in the config .


## Results and Models

| Architecture | Backbone  | Style   | Lr schd | Sampler | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:------------:|:---------:|:-------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| Faster R-CNN | R-50      | pytorch | 1x      |     Group Sampler    |  7.7   | -          | 51.6 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/openimages/faster_rcnn_r50_fpn_32x2_1x_openimages.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/openimages/faster_rcnn_r50_fpn_32x2_1x_openimages/faster_rcnn_r50_fpn_32x2_1x_openimages_20211130_231159-e87ab7ce.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/openimages/faster_rcnn_r50_fpn_32x2_1x_openimages/faster_rcnn_r50_fpn_32x2_1x_openimages_20211130_231159.log.json) |
| Faster R-CNN (Challenge 2019) | R-50  | pytorch | 1x |   Group Sampler  |  7.7  | -          | 54.5 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/openimages/faster_rcnn_r50_fpn_32x2_1x_openimages_challenge.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/openimages/faster_rcnn_r50_fpn_32x2_1x_openimages_challenge/faster_rcnn_r50_fpn_32x2_1x_openimages_challenge_20211229_071252-46380cde.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/openimages/faster_rcnn_r50_fpn_32x2_1x_openimages_challenge/faster_rcnn_r50_fpn_32x2_1x_openimages_challenge_20211229_071252.log.json) |
| Retinanet    | R-50      | pytorch | 1x      |    Group Sampler     |  6.6   | -          | 61.5 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/openimages/retinanet_r50_fpn_32x2_1x_openimages.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/openimages/retinanet_r50_fpn_32x2_1x_openimages/retinanet_r50_fpn_32x2_1x_openimages_20211223_071954-d2ae5462.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/openimages/retinanet_r50_fpn_32x2_1x_openimages/retinanet_r50_fpn_32x2_1x_openimages_20211223_071954.log.json) |
| SSD          | VGG16     | pytorch | 36e     |    Group Sampler     |  10.8  | -          | 35.4 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/openimages/ssd300_32x8_36e_openimages.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/openimages/ssd300_32x8_36e_openimages/ssd300_32x8_36e_openimages_20211224_000232-dce93846.pth) &#124; [log](ttps://download.openmmlab.com/mmdetection/v2.0/openimages/ssd300_32x8_36e_openimages/ssd300_32x8_36e_openimages_20211224_000232.log.json) |
