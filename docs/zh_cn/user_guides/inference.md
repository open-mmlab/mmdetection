# 使用已有模型在标准数据集上进行推理

MMDetection 提供了许多预训练好的检测模型，可以在 [Model Zoo](https://mmdetection.readthedocs.io/zh_CN/latest/model_zoo.html) 查看具体有哪些模型。

推理具体指使用训练好的模型来检测图像上的目标，本文将会展示具体步骤。

在 MMDetection 中，一个模型被定义为一个[配置文件](https://mmdetection.readthedocs.io/zh_CN/latest/user_guides/config.html) 和对应被存储在 checkpoint 文件内的模型参数的集合。

首先，我们建议从 [RTMDet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) 开始，其 [配置](https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py) 文件和 [checkpoint](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth) 文件在此。
我们建议将 checkpoint 文件下载到 `checkpoints` 文件夹内。

## 推理的高层编程接口

MMDetection 为在图片上推理提供了 Python 的高层编程接口。下面是建立模型和在图像或视频上进行推理的例子。

```python
import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector


# 指定模型的配置文件和 checkpoint 文件路径
config_file = 'configs/rtmdet/rtmdet_l_8xb32-300e_coco.py'
checkpoint_file = 'checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 初始化可视化工具
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# 从 checkpoint 中加载 Dataset_meta，并将其传递给模型的 init_detector
visualizer.dataset_meta = model.dataset_meta

# 测试单张图片并展示结果
img = 'test.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
result = inference_detector(model, img)

# 显示结果
img = mmcv.imread(img)
img = mmcv.imconvert(img, 'bgr', 'rgb')


visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    show=True)

# 测试视频并展示结果
# 构建测试 pipeline
model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

# 可视化工具在第33行和35行已经初完成了初始化，如果直接在一个 jupyter nodebook 中运行这个 demo，
# 这里则不需要再创建一个可视化工具了。
# 初始化可视化工具
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# 从 checkpoint 中加载 Dataset_meta，并将其传递给模型的 init_detector
visualizer.dataset_meta = model.dataset_meta

# 显示间隔 (ms), 0 表示暂停
wait_time = 1

video = mmcv.VideoReader('video.mp4')

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

Jupyter notebook 上的演示样例在 [demo/inference_demo.ipynb](https://github.com/open-mmlab/mmdetection/blob/main/demo/inference_demo.ipynb) 。

注意: `inference_detector` 目前仅支持单张图片的推理。

## 演示样例

我们还提供了三个演示脚本，它们是使用高层编程接口实现的。[源码在此](https://github.com/open-mmlab/mmdetection/blob/main/demo) 。

### 图片样例

这是在单张图片上进行推理的脚本。

```shell
python demo/image_demo.py \
    ${IMAGE_FILE} \
    ${CONFIG_FILE} \
    [--weights ${WEIGHTS}] \
    [--device ${GPU_ID}] \
    [--pred-score-thr ${SCORE_THR}]
```

运行样例：

```shell
python demo/image_demo.py demo/demo.jpg \
    configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    --weights checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
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
    configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth
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
    configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
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
    configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
    --nvdecode --out result.mp4
```

## 多模态算法的推理和验证

随着多模态视觉算法的不断发展，MMDetection 也完成了对这类算法的支持。这一小节我们通过 GLIP 算法和模型来演示如何使用对应多模态算法的 demo 和 eval 脚本。同时 MMDetection 也在 projects 下完成了 [gradio_demo 项目](../../../projects/gradio_demo/)，用户可以参照[文档](../../../projects/gradio_demo/README.md)在本地快速体验 MMDetection 中支持的各类图片输入的任务。

### 模型准备

首先需要安装多模态依赖：

```shell
# if source
pip install -r requirements/multimodal.txt

# if wheel
mim install mmdet[multimodal]
```

MMDetection 已经集成了 glip 算法和模型，可以直接使用链接下载使用：

```shell
cd mmdetection
wget https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth
```

### 推理演示

下载完成后我们就可以利用 `demo` 下的多模态推理脚本完成推理：

```shell
python demo/image_demo.py demo/demo.jpg glip_tiny_a_mmdet-b3654169.pth --texts bench
```

demo 效果如下图所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/234547841-266476c8-f987-4832-8642-34357be621c6.png" height="300"/>
</div>

如果想进行多种类型的识别，需要使用 `xx . xx .` 的格式在 `--texts` 字段后声明目标类型:

```shell
python demo/image_demo.py demo/demo.jpg glip_tiny_a_mmdet-b3654169.pth --texts 'bench . car .'
```

结果如下图所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/234548156-ef9bbc2e-7605-4867-abe6-048b8578893d.png" height="300"/>
</div>

推理脚本还支持输入一个句子作为 `--texts` 字段的输入：

```shell
python demo/image_demo.py demo/demo.jpg glip_tiny_a_mmdet-b3654169.pth --texts 'There are a lot of cars here.'
```

结果可以参考下图：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/234548490-d2e0a16d-1aad-4708-aea0-c829634fabbd.png" height="300"/>
</div>

### 验证演示

MMDetection 支持后的 GLIP 算法对比官方版本没有精度上的损失， benchmark 如下所示：

| Model                   | official mAP | mmdet mAP |
| ----------------------- | :----------: | :-------: |
| glip_A_Swin_T_O365.yaml |     42.9     |   43.0    |
| glip_Swin_T_O365.yaml   |     44.9     |   44.9    |
| glip_Swin_L.yaml        |     51.4     |   51.3    |

用户可以使用 `test.py` 脚本对模型精度进行验证，使用如下所示：

```shell
# 1 gpu
python tools/test.py configs/glip/glip_atss_swin-t_fpn_dyhead_pretrain_obj365.py glip_tiny_a_mmdet-b3654169.pth

# 8 GPU
./tools/dist_test.sh configs/glip/glip_atss_swin-t_fpn_dyhead_pretrain_obj365.py glip_tiny_a_mmdet-b3654169.pth 8
```

验证结果大致如下：

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
