# Inference with existing models

MMDetection provides hundreds of pretrained detection models in [Model Zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html).
This note will show how to inference, which means using trained models to detect objects on images.

In MMDetection, a model is defined by a [configuration file](config.md) and existing model parameters are save in a checkpoint file.

To start with, we recommend [Faster RCNN](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/configs/faster_rcnn) with this [configuration file](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py) and this [checkpoint file](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth). It is recommended to download the checkpoint file to `checkpoints` directory.

## High-level APIs for inference

MMDetection provide high-level Python APIs for inference on images. Here is an example of building the model and inference on given images or videos.

```python
import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules
from mmdet.apis import init_detector, inference_detector

# register all modules in mmdet into the registries
register_all_modules()

# Specify the path to model config and checkpoint file
config_file = 'configs/faster_rcnn/faster-rcnn_r50-fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta
# test a single image and show the results

img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)

# show the results
img = mmcv.imread(img)
img = mmcv.imconvert(img, 'bgr', 'rgb')


visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    show=True)

# test a video and show the results
# build test pipeline
model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

# init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

# The interval of show (s), 0 is block
wait_time = 1

video_reader = mmcv.VideoReader('video.mp4')

for frame in track_iter_progress(video_reader):
    result = inference_detector(model, frame, test_pipeline=test_pipeline)
    visualizer.add_datasample(
        name='video',
        image=frame,
        data_sample=result,
        draw_gt=False,
        show=False)
    frame = visualizer.get_image()

    cv2.namedWindow('video', 0)
    mmcv.imshow(frame, 'video', wait_time)

```

A notebook demo can be found in [demo/inference_demo.ipynb](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/demo/inference_demo.ipynb).

Note:  `inference_detector` only supports single-image inference for now.

## Asynchronous interface - supported for Python 3.7+

For Python 3.7+, MMDetection also supports async interfaces.
By utilizing CUDA streams, it allows not to block CPU on GPU bound inference code and enables better CPU/GPU utilization for single-threaded application. Inference can be done concurrently either between different input data samples or between different models of some inference pipeline.

See `tests/async_benchmark.py` to compare the speed of synchronous and asynchronous interfaces.

```python
import asyncio
import torch
from mmdet.apis import init_detector, async_inference_detector
from mmdet.registry import VISUALIZERS
from mmdet.utils.contextmanagers import concurrent

async def main():
    config_file = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # test a single image and show the results
    img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once

    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, img))
    result = await asyncio.gather(tasks)
    # show the results
    img = mmcv.imread(img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result[0],
        draw_gt=False,
        show=True,
        wait_time=0)

asyncio.run(main())
```

## Demos

We also provide three demo scripts, implemented with high-level APIs and supporting functionality codes.
Source codes are available [here](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/demo).

### Image demo

This script performs inference on a single image.

```shell
python demo/image_demo.py \
    ${IMAGE_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--score-thr ${SCORE_THR}]
```

Examples:

```shell
python demo/image_demo.py demo/demo.jpg \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
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
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
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
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
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
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --nvdecode --out result.mp4
```
