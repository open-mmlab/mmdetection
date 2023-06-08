# Inference with existing models

MMDetection provides hundreds of pre-trained detection models in [Model Zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html).
This note will show how to inference, which means using trained models to detect objects on images.

In MMDetection, a model is defined by a [configuration file](https://mmdetection.readthedocs.io/en/latest/user_guides/config.html) and existing model parameters are saved in a checkpoint file.

To start with, we recommend [RTMDet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) with this [configuration file](https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py) and this [checkpoint file](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth). It is recommended to download the checkpoint file to `checkpoints` directory.

## High-level APIs for inference

MMDetection provides high-level Python APIs for inference on images. Here is an example of building the model and inference on given images or videos.

```python
import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector


# Specify the path to model config and checkpoint file
config_file = 'configs/rtmdet/rtmdet_l_8xb32-300e_coco.py'
checkpoint_file = 'checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# The dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

# Test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)

# Show the results
img = mmcv.imread(img)
img = mmcv.imconvert(img, 'bgr', 'rgb')


visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    show=True)

# Test a video and show the results
# Build test pipeline
model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

# visualizer has been created in line 31 and 34, if you run this demo in one notebook,
# you need not build the visualizer again.

# Init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# The dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

# The interval of show (ms), 0 is block
wait_time = 1

video_reader = mmcv.VideoReader('video.mp4')

cv2.namedWindow('video', 0)

for frame in track_iter_progress(video_reader):
    result = inference_detector(model, frame, test_pipeline=test_pipeline)
    visualizer.add_datasample(
        name='video',
        image=frame,
        data_sample=result,
        draw_gt=False,
        show=False)
    frame = visualizer.get_image()
    mmcv.imshow(frame, 'video', wait_time)

cv2.destroyAllWindows()
```

A notebook demo can be found in [demo/inference_demo.ipynb](https://github.com/open-mmlab/mmdetection/blob/main/demo/inference_demo.ipynb).

Note:  `inference_detector` only supports single-image inference for now.

## Demos

We also provide three demo scripts, implemented with high-level APIs and supporting functionality codes.
Source codes are available [here](https://github.com/open-mmlab/mmdetection/blob/main/demo).

### Image demo

This script performs inference on a single image.

```shell
python demo/image_demo.py \
    ${IMAGE_FILE} \
    ${CONFIG_FILE} \
    [--weights ${WEIGHTS}] \
    [--device ${GPU_ID}] \
    [--pred-score-thr ${SCORE_THR}]
```

Examples:

```shell
python demo/image_demo.py demo/demo.jpg \
    configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    --weights checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
    --device cpu
```

#### Webcam demo

This is a live demo from a webcam.

```shell
python demo/webcam_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--camera-id ${CAMERA-ID}] \
    [--score-thr ${SCORE_THR}]
```

Examples:

```shell
python demo/webcam_demo.py \
    configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth
```

### Video demo

This script performs inference on a video.

```shell
python demo/video_demo.py \
    ${VIDEO_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--score-thr ${SCORE_THR}] \
    [--out ${OUT_FILE}] \
    [--show] \
    [--wait-time ${WAIT_TIME}]
```

Examples:

```shell
python demo/video_demo.py demo/demo.mp4 \
    configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
    --out result.mp4
```

#### Video demo with GPU acceleration

This script performs inference on a video with GPU acceleration.

```shell
python demo/video_gpuaccel_demo.py \
    ${VIDEO_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--score-thr ${SCORE_THR}] \
    [--nvdecode] \
    [--out ${OUT_FILE}] \
    [--show] \
    [--wait-time ${WAIT_TIME}]
```

Examples:

```shell
python demo/video_gpuaccel_demo.py demo/demo.mp4 \
    configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
    --nvdecode --out result.mp4
```

## Multi-modal algorithm inference demo and evaluation

Thanks to the tremendous amount of work of different multi-modality researches, MMDetection has also supported some of them. In this section, we use GLIP as an example to show how users can run inference and evaluation on these algorithms.

### Preparation

MMDetection has already implemented some model converter scripts, so we can directly download the pre-trained GLIP models from the official repository. The code is as follows:

```shell
cd mmdetection
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_a_tiny_o365.pth

python tools/model_converters/glip_to_mmdet.py --src glip_a_tiny_o365.pth --dst glip_tiny_mmdet.pth
```

### Inference

Once the model is successfully converted to the MMDetction format, users can use the `multimodal_demo.py` script to run inferences.

```shell
python demo/multimodal_demo.py demo/demo.jpg bench configs/glip/glip_atss_swin-t_fpn_dyhead_pretrain_obj365.py glip_tiny_mmdet.pth
```

Demo result will be similar to this:

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/234548156-ef9bbc2e-7605-4867-abe6-048b8578893d.png" height="300"/>
</div>

If users would like to detect multiple targets, please declare in the format of \`"xx . xx ."':

```shell
python demo/multimodal_demo.py demo/demo.jpg "bench . car . " configs/glip/glip_atss_swin-t_fpn_dyhead_pretrain_obj365.py glip_tiny_mmdet.pth
```

And the result will be like this one:

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/234548490-d2e0a16d-1aad-4708-aea0-c829634fabbd.png" height="300"/>
</div>

### Evaluation

The GLIP implementation in MMDetection does not have any performance degradation, our benchmark is as follows:

| Model                   | official mAP | mmdet mAP |
| ----------------------- | :----------: | :-------: |
| glip_A_Swin_T_O365.yaml |     42.9     |   43.0    |
| glip_Swin_T_O365.yaml   |     44.9     |   44.9    |
| glip_Swin_L.yaml        |     51.4     |   51.3    |

Users can use the test script we provided to run evaluation as well. Here is a basic example:

```shell
# 1 gpu
python tools/test.py configs/glip/glip_atss_swin-t_fpn_dyhead_pretrain_obj365.py glip_tiny_mmdet.pth

# 8 GPU
./tools/dist_test.sh configs/glip/glip_atss_swin-t_fpn_dyhead_pretrain_obj365.py glip_tiny_mmdet.pth 8
```

The result will be similar to this:

```shell
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.428
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.594
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.466
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.300
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.477
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.534
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.634
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.634
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.634
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.473
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.690
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.789
```
