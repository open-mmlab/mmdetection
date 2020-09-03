## Common settings

- All baselines were trained using 8 GPU with a batch size of 8 (1 images per GPU) using the [linear scaling rule](https://arxiv.org/abs/1706.02677) to scale the learning rate.
- All models were trained on `cityscapes_train`, and tested on `cityscapes_val`.
- 1x training schedule indicates 64 epochs which corresponds to slightly less than the 24k iterations reported in the original schedule from the [Mask R-CNN paper](https://arxiv.org/abs/1703.06870)
- COCO pre-trained weights are used to initialize.
- A conversion [script](../../tools/convert_datasets/cityscapes.py) is provided to convert Cityscapes into COCO format. Please refer to [install.md](../../docs/install.md#prepare-datasets) for details.
- `CityscapesDataset` implemented three evaluation methods. `bbox` and `segm` are standard COCO bbox/mask AP. `cityscapes` is the cityscapes dataset official evaluation, which may be slightly higher than COCO.


### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Scale    | Mem (GB) | Inf time (fps) | box AP | Download   |
| :-------------: | :-----: | :-----: | :---:    | :------: | :------------: | :----: | :--------: |
|    R-50-FPN     | pytorch |   1x    | 800-1024 |   5.2    |       -        |  40.3  | [model](http://download.openmmlab.com/mmdetection/v2.0/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_20200502_114915.log.json) |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Scale    | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------: | :------------: | :----: | :-----: | :------: |
|    R-50-FPN     | pytorch |   1x    | 800-1024 |   5.3    |       -        |  41.0  |  35.8   | [model](http://download.openmmlab.com/mmdetection/v2.0/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes_20200502-6ea77f0e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes_20200502_114915.log.json) |
