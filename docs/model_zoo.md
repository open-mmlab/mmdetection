# Benchmark and Model Zoo

## Mirror sites

We use AWS as the main site to host our model zoo, and maintain a mirror on aliyun.
You can replace `https://s3.ap-northeast-2.amazonaws.com/open-mmlab` with `https://open-mmlab.oss-cn-beijing.aliyuncs.com` in model urls.

## Common settings

- All FPN baselines and RPN-C4 baselines were trained using 8 GPU with a batch size of 16 (2 images per GPU). Other C4 baselines were trained using 8 GPU with a batch size of 8 (1 image per GPU).
- All models were trained on `coco_2017_train`, and tested on the `coco_2017_val`.
- We use distributed training and BN layer stats are fixed.
- We adopt the same training schedules as Detectron. 1x indicates 12 epochs and 2x indicates 24 epochs, which corresponds to slightly less iterations than Detectron and the difference can be ignored.
- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo.
- For fair comparison with other codebases, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 8 GPUs. Note that this value is usually less than what `nvidia-smi` shows.
- We report the inference time as the overall time including data loading, network forwarding and post processing.


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

### Other datasets

We also benchmark some methods on [PASCAL VOC](https://github.com/open-mmlab/mmdetection/blob/master/configs/pascal_voc), [Cityscapes](https://github.com/open-mmlab/mmdetection/blob/master/configs/cityscapes) and [WIDER FACE](https://github.com/open-mmlab/mmdetection/blob/master/configs/wider_face).


## Speed benchmark
We compare the training speed of Mask R-CNN with some other popular frameworks (The data is copied from [detectron2](https://github.com/facebookresearch/detectron2/blob/master/docs/notes/benchmarks.md)).

| Implementation       | Throughput (img/s) |
|----------------------|--------------------|
| [Detectron2](https://github.com/facebookresearch/detectron2) | 61 |
| [MMDetection](https://github.com/open-mmlab/mmdetection) | 60 |
| [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/)   | 51 |
| [tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) | 50 |
| [simpledet](https://github.com/TuSimple/simpledet/) | 39 |
| [Detectron](https://github.com/facebookresearch/Detectron) | 19 |
| [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN/) | 14 |

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

<table border="1">
  <tr>
    <th>Type</th>
    <th>Lr schd</th>
    <th>Detectron2</th>
    <th>mmdetection</th>
  </tr>
  <tr>
    <td rowspan="2">Faster R-CNN</td>
    <td>1x</td>
    <td>37.9</td>
    <td>38.0</td>
  </tr>
  <tr>
    <td>3x</td>
    <td>40.2</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">Mask R-CNN</td>
    <td>1x</td>
    <td>38.6 &amp; 35.2</td>
    <td>38.8 &amp; 35.4</td>
  </tr>
  <tr>
    <td>3x</td>
    <td>41.0 &amp; 37.2 </td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">Retinanet</td>
    <td>1x</td>
    <td>36.5</td>
    <td>37.0</td>
  </tr>
  <tr>
    <td>3x</td>
    <td>37.9</td>
    <td>-</td>
  </tr>
</table>

### Training Speed

The training speed is measure with s/iter. The lower, the better.

<table border="1">
  <tr>
    <th>Type</th>
    <th>Detectron2</th>
    <th>mmdetection</th>
  </tr>
  <tr>
    <td>Faster R-CNN</td>
    <td>0.210</td>
    <td>0.216</td>
  </tr>
  <tr>
    <td>Mask R-CNN</td>
    <td>0.261</td>
    <td>0.265</td>
  </tr>
  <tr>
    <td>Retinanet</td>
    <td>0.200</td>
    <td>0.205</td>
  </tr>
</table>


### Inference Speed

The inference speed is measured with fps (img/s) on a single GPU, the higher, the better.
To be consistent with Detectron2, we report the pure inference speed (without the time of data loading).
For Mask R-CNN, we exclude the time of RLE encoding in post-processing.
We also include the officially reported speed in the parentheses, which is slightly higher
than the results tested on our server due to differences of hardwares.

<table border="1">
  <tr>
    <th>Type</th>
    <th>Detectron2</th>
    <th>mmdetection</th>
  </tr>
  <tr>
    <td>Faster R-CNN</td>
    <td>25.6 (26.3)</td>
    <td>22.2</td>
  </tr>
  <tr>
    <td>Mask R-CNN</td>
    <td>22.5 (23.3)</td>
    <td>19.6</td>
  </tr>
  <tr>
    <td>Retinanet</td>
    <td>17.8 (18.2)</td>
    <td>20.6</td>
  </tr>
</table>

### Training memory

<table border="1">
  <tr>
    <th>Type</th>
    <th>Detectron2</th>
    <th>mmdetection</th>
  </tr>
  <tr>
    <td>Faster R-CNN</td>
    <td>3.0</td>
    <td>3.8</td>
  </tr>
  <tr>
    <td>Mask R-CNN</td>
    <td>3.4</td>
    <td>3.9</td>
  </tr>
  <tr>
    <td>Retinanet</td>
    <td>3.9</td>
    <td>3.4</td>
  </tr>
</table>
