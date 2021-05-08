# Benchmark and Model Zoo

## Mirror sites

We use AWS as the main site to host our model zoo, and maintain a mirror on aliyun.
You can replace `https://s3.ap-northeast-2.amazonaws.com/open-mmlab` with `https://open-mmlab.oss-cn-beijing.aliyuncs.com` in model urls.

## Common settings

- All models were trained on `coco_2017_train`, and tested on the `coco_2017_val`.
- We use distributed training.
- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo, caffe-style pretrained backbones are converted from the newly released model from detectron2.
- For fair comparison with other codebases, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 8 GPUs. Note that this value is usually less than what `nvidia-smi` shows.
- We report the inference time as the total time of network forwarding and post-processing, excluding the data loading time. Results are obtained with the script [benchmark.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/analysis_tools/benchmark.py) which computes the average time on 2000 images.

## Baselines

### RPN

Please refer to [RPN](https://github.com/open-mmlab/mmdetection/blob/master/configs/rpn) for details.

### Faster R-CNN

Please refer to [Faster R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn) for details.

### Mask R-CNN

Please refer to [Mask R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn) for details.

### Fast R-CNN (with pre-computed proposals)

Please refer to [Fast R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/fast_rcnn) for details.

### RetinaNet

Please refer to [RetinaNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet) for details.

### Cascade R-CNN and Cascade Mask R-CNN

Please refer to [Cascade R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/cascade_rcnn) for details.

### Hybrid Task Cascade (HTC)

Please refer to [HTC](https://github.com/open-mmlab/mmdetection/blob/master/configs/htc) for details.

### SSD

Please refer to [SSD](https://github.com/open-mmlab/mmdetection/blob/master/configs/ssd) for details.

### Group Normalization (GN)

Please refer to [Group Normalization](https://github.com/open-mmlab/mmdetection/blob/master/configs/gn) for details.

### Weight Standardization

Please refer to [Weight Standardization](https://github.com/open-mmlab/mmdetection/blob/master/configs/gn+ws) for details.

### Deformable Convolution v2

Please refer to [Deformable Convolutional Networks](https://github.com/open-mmlab/mmdetection/blob/master/configs/dcn) for details.

### CARAFE: Content-Aware ReAssembly of FEatures

Please refer to [CARAFE](https://github.com/open-mmlab/mmdetection/blob/master/configs/carafe) for details.

### Instaboost

Please refer to [Instaboost](https://github.com/open-mmlab/mmdetection/blob/master/configs/instaboost) for details.

### Libra R-CNN

Please refer to [Libra R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/libra_rcnn) for details.

### Guided Anchoring

Please refer to [Guided Anchoring](https://github.com/open-mmlab/mmdetection/blob/master/configs/guided_anchoring) for details.

### FCOS

Please refer to [FCOS](https://github.com/open-mmlab/mmdetection/blob/master/configs/fcos) for details.

### FoveaBox

Please refer to [FoveaBox](https://github.com/open-mmlab/mmdetection/blob/master/configs/foveabox) for details.

### RepPoints

Please refer to [RepPoints](https://github.com/open-mmlab/mmdetection/blob/master/configs/reppoints) for details.

### FreeAnchor

Please refer to [FreeAnchor](https://github.com/open-mmlab/mmdetection/blob/master/configs/free_anchor) for details.

### Grid R-CNN (plus)

Please refer to [Grid R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/grid_rcnn) for details.

### GHM

Please refer to [GHM](https://github.com/open-mmlab/mmdetection/blob/master/configs/ghm) for details.

### GCNet

Please refer to [GCNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/gcnet) for details.

### HRNet

Please refer to [HRNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/hrnet) for details.

### Mask Scoring R-CNN

Please refer to [Mask Scoring R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/ms_rcnn) for details.

### Train from Scratch

Please refer to [Rethinking ImageNet Pre-training](https://github.com/open-mmlab/mmdetection/blob/master/configs/scratch) for details.

### NAS-FPN

Please refer to [NAS-FPN](https://github.com/open-mmlab/mmdetection/blob/master/configs/nas_fpn) for details.

### ATSS

Please refer to [ATSS](https://github.com/open-mmlab/mmdetection/blob/master/configs/atss) for details.

### FSAF

Please refer to [FSAF](https://github.com/open-mmlab/mmdetection/blob/master/configs/fsaf) for details.

### RegNetX

Please refer to [RegNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/regnet) for details.

### Res2Net

Please refer to [Res2Net](https://github.com/open-mmlab/mmdetection/blob/master/configs/res2net) for details.

### GRoIE

Please refer to [GRoIE](https://github.com/open-mmlab/mmdetection/blob/master/configs/groie) for details.

### Dynamic R-CNN

Please refer to [Dynamic R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/dynamic_rcnn) for details.

### PointRend

Please refer to [PointRend](https://github.com/open-mmlab/mmdetection/blob/master/configs/point_rend) for details.

### DetectoRS

Please refer to [DetectoRS](https://github.com/open-mmlab/mmdetection/blob/master/configs/detectors) for details.

### Generalized Focal Loss

Please refer to [Generalized Focal Loss](https://github.com/open-mmlab/mmdetection/blob/master/configs/gfl) for details.

### CornerNet

Please refer to [CornerNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/cornernet) for details.

### YOLOv3

Please refer to [YOLOv3](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolo) for details.

### PAA

Please refer to [PAA](https://github.com/open-mmlab/mmdetection/blob/master/configs/paa) for details.

### SABL

Please refer to [SABL](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl) for details.

### CentripetalNet

Please refer to [CentripetalNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/centripetalnet) for details.

### ResNeSt

Please refer to [ResNeSt](https://github.com/open-mmlab/mmdetection/blob/master/configs/resnest) for details.

### DETR

Please refer to [DETR](https://github.com/open-mmlab/mmdetection/blob/master/configs/detr) for details.

### Deformable DETR

Please refer to [Deformable DETR](https://github.com/open-mmlab/mmdetection/blob/master/configs/deformable_detr) for details.

### AutoAssign

Please refer to [AutoAssign](https://github.com/open-mmlab/mmdetection/blob/master/configs/autoassign) for details.

### YOLOF

Please refer to [YOLOF](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolof) for details.

### Other datasets

We also benchmark some methods on [PASCAL VOC](https://github.com/open-mmlab/mmdetection/blob/master/configs/pascal_voc), [Cityscapes](https://github.com/open-mmlab/mmdetection/blob/master/configs/cityscapes) and [WIDER FACE](https://github.com/open-mmlab/mmdetection/blob/master/configs/wider_face).

### Pre-trained Models

We also train [Faster R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn) and [Mask R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn) using ResNet-50 and [RegNetX-3.2G](https://github.com/open-mmlab/mmdetection/blob/master/configs/regnet) with multi-scale training and longer schedules. These models serve as strong pre-trained models for downstream tasks for convenience.

## Speed benchmark

### Training Speed benchmark

We provide [analyze_logs.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/analysis_tools/analyze_logs.py) to get average time of iteration in training. You can find examples in [Log Analysis](https://mmdetection.readthedocs.io/en/latest/useful_tools.html#log-analysis).

We compare the training speed of Mask R-CNN with some other popular frameworks (The data is copied from [detectron2](https://github.com/facebookresearch/detectron2/blob/master/docs/notes/benchmarks.md)).
For mmdetection, we benchmark with [mask_rcnn_r50_caffe_fpn_poly_1x_coco_v1.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_poly_1x_coco_v1.py), which should have the same setting with [mask_rcnn_R_50_FPN_noaug_1x.yaml](https://github.com/facebookresearch/detectron2/blob/master/configs/Detectron1-Comparisons/mask_rcnn_R_50_FPN_noaug_1x.yaml) of detectron2.
We also provide the [checkpoint](http://download.openmmlab.com/mmdetection/v2.0/benchmark/mask_rcnn_r50_caffe_fpn_poly_1x_coco_no_aug/mask_rcnn_r50_caffe_fpn_poly_1x_coco_no_aug_compare_20200518-10127928.pth) and [training log](http://download.openmmlab.com/mmdetection/v2.0/benchmark/mask_rcnn_r50_caffe_fpn_poly_1x_coco_no_aug/mask_rcnn_r50_caffe_fpn_poly_1x_coco_no_aug_20200518_105755.log.json) for reference. The throughput is computed as the average throughput in iterations 100-500 to skip GPU warmup time.

| Implementation       | Throughput (img/s) |
|----------------------|--------------------|
| [Detectron2](https://github.com/facebookresearch/detectron2) | 62 |
| [MMDetection](https://github.com/open-mmlab/mmdetection) | 61 |
| [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/)   | 53 |
| [tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) | 50 |
| [simpledet](https://github.com/TuSimple/simpledet/) | 39 |
| [Detectron](https://github.com/facebookresearch/Detectron) | 19 |
| [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN/) | 14 |

### Inference Speed Benchmark

We provide [benchmark.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/analysis_tools/benchmark.py) to benchmark the inference latency.
The script benchmarkes the model with 2000 images and calculates the average time ignoring first 5 times. You can change the output log interval (defaults: 50) by setting `LOG-INTERVAL`.

```shell
python toools/benchmark.py ${CONFIG} ${CHECKPOINT} [--log-interval $[LOG-INTERVAL]] [--fuse-conv-bn]
```

The latency of all models in our model zoo is benchmarked without setting `fuse-conv-bn`, you can get a lower latency by setting it.

## Comparison with Detectron2

We compare mmdetection with [Detectron2](https://github.com/facebookresearch/detectron2.git) in terms of speed and performance.
We use the commit id [185c27e](https://github.com/facebookresearch/detectron2/tree/185c27e4b4d2d4c68b5627b3765420c6d7f5a659)(30/4/2020) of detectron.
For fair comparison, we install and run both frameworks on the same machine.

### Hardware

- 8 NVIDIA Tesla V100 (32G) GPUs
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

### Software environment

- Python 3.7
- PyTorch 1.4
- CUDA 10.1
- CUDNN 7.6.03
- NCCL 2.4.08

### Performance

| Type         | Lr schd | Detectron2  | mmdetection | Download |
|--------------|---------|-------------|-------------|-------------|
| [Faster R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py) | 1x      | [37.9](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml)        | 38.0        | [model](http://download.openmmlab.com/mmdetection/v2.0/benchmark/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-5324cff8.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/benchmark/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco_20200429_234554.log.json) |
| [Mask R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py)   | 1x      | [38.6 & 35.2](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml) | 38.8 & 35.4 | [model](http://download.openmmlab.com/mmdetection/v2.0/benchmark/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco-dbecf295.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/benchmark/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco_20200430_054239.log.json) |
| [Retinanet](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet/retinanet_r50_caffe_fpn_mstrain_1x_coco.py)    | 1x      | [36.5](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml)        | 37.0        | [model](http://download.openmmlab.com/mmdetection/v2.0/benchmark/retinanet_r50_caffe_fpn_mstrain_1x_coco/retinanet_r50_caffe_fpn_mstrain_1x_coco-586977a0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/benchmark/retinanet_r50_caffe_fpn_mstrain_1x_coco/retinanet_r50_caffe_fpn_mstrain_1x_coco_20200430_014748.log.json) |

### Training Speed

The training speed is measure with s/iter. The lower, the better.

| Type         | Detectron2 | mmdetection |
|--------------|------------|-------------|
| Faster R-CNN | 0.210      | 0.216       |
| Mask R-CNN   | 0.261      | 0.265       |
| Retinanet    | 0.200      | 0.205       |

### Inference Speed

The inference speed is measured with fps (img/s) on a single GPU, the higher, the better.
To be consistent with Detectron2, we report the pure inference speed (without the time of data loading).
For Mask R-CNN, we exclude the time of RLE encoding in post-processing.
We also include the officially reported speed in the parentheses, which is slightly higher
than the results tested on our server due to differences of hardwares.

| Type         | Detectron2  | mmdetection |
|--------------|-------------|-------------|
| Faster R-CNN | 25.6 (26.3) | 22.2        |
| Mask R-CNN   | 22.5 (23.3) | 19.6        |
| Retinanet    | 17.8 (18.2) | 20.6        |

### Training memory

| Type         | Detectron2 | mmdetection |
|--------------|------------|-------------|
| Faster R-CNN | 3.0        | 3.8         |
| Mask R-CNN   | 3.4        | 3.9         |
| Retinanet    | 3.9        | 3.4         |
