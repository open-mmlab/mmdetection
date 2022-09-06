# 使用已有模型在标准数据集上进行推理（待更新）

推理是指使用训练好的模型来检测图像上的目标。在 MMDetection 中，一个模型被定义为一个配置文件和对应的存储在 checkpoint 文件内的模型参数的集合。

首先，我们建议从 [Faster RCNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn) 开始，其 [配置](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) 文件和 [checkpoint](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) 文件在此。
我们建议将 checkpoint 文件下载到 `checkpoints` 文件夹内。

## 推理的高层编程接口

MMDetection 为在图片上推理提供了 Python 的高层编程接口。下面是建立模型和在图像或视频上进行推理的例子。

```python
from mmdet.apis import init_detector, inference_detector
import mmcv

# 指定模型的配置文件和 checkpoint 文件路径
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并展示结果
img = 'test.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
result = inference_detector(model, img)
# 在一个新的窗口中将结果可视化
model.show_result(img, result)
# 或者将可视化结果保存为图片
model.show_result(img, result, out_file='result.jpg')

# 测试视频并展示结果
video = mmcv.VideoReader('video.mp4')
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1)
```

jupyter notebook 上的演示样例在 [demo/inference_demo.ipynb](https://github.com/open-mmlab/mmdetection/blob/master/demo/inference_demo.ipynb) 。

## 异步接口-支持 Python 3.7+

对于 Python 3.7+，MMDetection 也有异步接口。利用 CUDA 流，绑定 GPU 的推理代码不会阻塞 CPU，从而使得 CPU/GPU 在单线程应用中能达到更高的利用率。在推理流程中，不同数据样本的推理和不同模型的推理都能并发地运行。

您可以参考 `tests/async_benchmark.py` 来对比同步接口和异步接口的运行速度。

```python
import asyncio
import torch
from mmdet.apis import init_detector, async_inference_detector
from mmdet.utils.contextmanagers import concurrent

async def main():
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    # 此队列用于并行推理多张图像
    streamqueue = asyncio.Queue()
    # 队列大小定义了并行的数量
    streamqueue_size = 3

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))

    # 测试单张图片并展示结果
    img = 'test.jpg'  # or 或者 img = mmcv.imread(img)，这样图片仅会被读一次

    async with concurrent(streamqueue):
        result = await async_inference_detector(model, img)

    # 在一个新的窗口中将结果可视化
    model.show_result(img, result)
    # 或者将可视化结果保存为图片
    model.show_result(img, result, out_file='result.jpg')


asyncio.run(main())

```

## 演示样例

我们还提供了三个演示脚本，它们是使用高层编程接口实现的。 [源码在此](https://github.com/open-mmlab/mmdetection/tree/master/demo) 。

### 图片样例

这是在单张图片上进行推理的脚本，可以开启 `--async-test` 来进行异步推理。

```shell
python demo/image_demo.py \
    ${IMAGE_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--score-thr ${SCORE_THR}] \
    [--async-test]
```

运行样例：

```shell
python demo/image_demo.py demo/demo.jpg \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --device cpu
```

### 摄像头样例

这是使用摄像头实时图片的推理脚本。

```shell
python demo/webcam_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--camera-id ${CAMERA-ID}] \
    [--score-thr ${SCORE_THR}]
```

运行样例：

```shell
python demo/webcam_demo.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
```

### 视频样例

这是在视频样例上进行推理的脚本。

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

运行样例：

```shell
python demo/video_demo.py demo/demo.mp4 \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --out result.mp4
```

### 视频样例，显卡加速版本

这是在视频样例上进行推理的脚本，使用显卡加速。

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

运行样例：

```shell
python demo/video_gpuaccel_demo.py demo/demo.mp4 \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --nvdecode --out result.mp4
```
