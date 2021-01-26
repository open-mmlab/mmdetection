# DeepFashion

[DATASET]

[MMFashion](https://github.com/open-mmlab/mmfashion) develops "fashion parsing and segmentation" module
based on the dataset
[DeepFashion-Inshop](https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E?usp=sharing).
Its annotation follows COCO style.
To use it, you need to first download the data. Note that we only use "img_highres" in this task.
The file tree should be like this:

```sh
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── DeepFashion
│   │   ├── In-shop
│   │   ├── Anno
│   │   │   ├── segmentation
│   │   │   |   ├── DeepFashion_segmentation_train.json
│   │   │   |   ├── DeepFashion_segmentation_query.json
│   │   │   |   ├── DeepFashion_segmentation_gallery.json
│   │   │   ├── list_bbox_inshop.txt
│   │   │   ├── list_description_inshop.json
│   │   │   ├── list_item_inshop.txt
│   │   │   └── list_landmarks_inshop.txt
│   │   ├── Eval
│   │   │   └── list_eval_partition.txt
│   │   ├── Img
│   │   │   ├── img
│   │   │   │   ├──XXX.jpg
│   │   │   ├── img_highres
│   │   │   └── ├──XXX.jpg

```

After that you can train the Mask RCNN r50 on DeepFashion-In-shop dataset by launching training with the `mask_rcnn_r50_fpn_1x.py` config
or creating your own config file.

```
@inproceedings{liuLQWTcvpr16DeepFashion,
   author = {Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou},
   title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
   booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
   month = {June},
   year = {2016}
}
```

## Model Zoo

|   Backbone  |  Model type  |       Dataset       |  bbox detection Average Precision  | segmentation Average Precision |  Config |      Download (Google)      |
| :---------: | :----------: | :-----------------: | :--------------------------------: | :----------------------------: | :---------:| :-------------------------: |
|   ResNet50  |   Mask RCNN  | DeepFashion-In-shop |                0.599               |              0.584             |[config](https://github.com/open-mmlab/mmdetection/blob/master/configs/deepfashion/mask_rcnn_r50_fpn_15e_deepfashion.py)|  [model](https://drive.google.com/open?id=1q6zF7J6Gb-FFgM87oIORIt6uBozaXp5r) &#124; [log](https://drive.google.com/file/d/1qTK4Dr4FFLa9fkdI6UVko408gkrfTRLP/view?usp=sharing)   |
