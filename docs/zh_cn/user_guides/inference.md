# 使用已有模型在标准数据集上进行推理

MMDetection 提供了许多预训练好的检测模型，可以在 [Model Zoo](https://mmdetection.readthedocs.io/zh_CN/latest/model_zoo.html) 查看具体有哪些模型。

推理具体指使用训练好的模型来检测图像上的目标，本文将会展示具体步骤。

在 MMDetection 中，一个模型被定义为一个[配置文件](https://mmdetection.readthedocs.io/zh_CN/latest/user_guides/config.html) 和对应被存储在 checkpoint 文件内的模型参数的集合。

首先，我们建议从 [RTMDet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) 开始，其 [配置](https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py) 文件和 [checkpoint](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth) 文件在此。
我们建议将 checkpoint 文件下载到 `checkpoints` 文件夹内。

## 推理的高层编程接口——推理器

在 OpenMMLab 中，所有的推理操作都被统一到了推理器 `Inferencer` 中。推理器被设计成为一个简洁易用的 API，它在不同的 OpenMMLab 库中都有着非常相似的接口。
下面介绍的演示样例都放在 [demo/inference_demo.ipynb](https://github.com/open-mmlab/mmdetection/blob/main/demo/inference_demo.ipynb) 中方便大家尝试。

### 基础用法

使用 `DetInferencer`，您只需 3 行代码就可以获得推理结果。

```python
from mmdet.apis import DetInferencer

# 初始化模型
inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco')

# 推理示例图片
inferencer('demo/demo.jpg', show=True)
```

可视化结果将被显示在一个新窗口中：

<div align="center">
    <img src='https://github.com/open-mmlab/mmdetection/assets/27466624/311df42d-640a-4a5b-9ad9-9ba7f3ec3a2f' />
</div>

```{note}
如果你在没有 GUI 的服务器上，或者通过禁用 X11 转发的 SSH 隧道运行以上命令，`show` 选项将不起作用。然而，你仍然可以通过设置 `out_dir` 参数将可视化数据保存到文件。阅读 [储存结果](#储存结果) 了解详情。
```

### 初始化

每个推理器必须使用一个模型进行初始化。初始化时，可以手动选择推理设备。

#### 模型初始化

- 要用 MMDetection 的预训练模型进行推理，只需要把它的名字传给参数 `model`，权重将自动从 OpenMMLab 的模型库中下载和加载。

  ```python
  inferencer = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco')
  ```

  在 MMDetection 中有一个非常容易的方法，可以列出所有模型名称。

  ```python
  # models 是一个模型名称列表，它们将自动打印
  models = DetInferencer.list_models('mmdet')
  ```

  你可以通过将权重的路径或 URL 传递给 `weights` 来让推理器加载自定义的权重。

  ```python
  inferencer = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco', weights='path/to/rtmdet.pth')
  ```

- 要加载自定义的配置和权重，你可以把配置文件的路径传给 `model`，把权重的路径传给 `weights`。

  ```python
  inferencer = DetInferencer(model='path/to/rtmdet_config.py', weights='path/to/rtmdet.pth')
  ```

- 默认情况下，[MMEngine](https://github.com/open-mmlab/mmengine/) 会在训练模型时自动将配置文件转储到权重文件中。如果你有一个在 MMEngine 上训练的权重，你也可以将权重文件的路径传递给 `weights`，而不需要指定 `model`：

  ```python
  # 如果无法在权重中找到配置文件，则会引发错误。目前 MMDetection 模型库中只有 ddq-detr-4scale_r50 的权重可以这样加载。
  inferencer = DetInferencer(weights='https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq-detr-4scale_r50_8xb2-12e_coco/ddq-detr-4scale_r50_8xb2-12e_coco_20230809_170711-42528127.pth')
  ```

- 传递配置文件到 `model` 而不指定 `weights` 则会产生一个随机初始化的模型。

#### 推理设备

每个推理器实例都会跟一个设备绑定。默认情况下，最佳设备是由 [MMEngine](https://github.com/open-mmlab/mmengine/) 自动决定的。你也可以通过指定 `device` 参数来改变设备。例如，你可以使用以下代码在 GPU 1 上创建一个推理器。

```python
inferencer = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco', device='cuda:1')
```

如要在 CPU 上创建一个推理器：

```python
inferencer = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco', device='cpu')
```

请参考 [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) 了解 `device` 参数支持的所有形式。

### 推理

当推理器初始化后，你可以直接传入要推理的原始数据，从返回值中获取推理结果。

#### 输入

输入可以是以下任意一种格式：

- str: 图像的路径/URL。

  ```python
  inferencer('demo/demo.jpg')
  ```

- array: 图像的 numpy 数组。它应该是 BGR 格式。

  ```python
  import mmcv
  array = mmcv.imread('demo/demo.jpg')
  inferencer(array)
  ```

- list: 基本类型的列表。列表中的每个元素都将单独处理。

  ```python
  inferencer(['img_1.jpg', 'img_2.jpg])
  # 列表内混合类型也是允许的
  inferencer(['img_1.jpg', array])
  ```

- str: 目录的路径。目录中的所有图像都将被处理。

  ```python
  inferencer('path/to/your_imgs/')
  ```

#### 输出

默认情况下，每个推理器都以字典格式返回预测结果。

- `visualization` 包含可视化的预测结果。但默认情况下，它是一个空列表，除非 `return_vis=True`。

- `predictions` 包含以 json-可序列化格式返回的预测结果。

```python
{
      'predictions' : [
        # 每个实例都对应于一个输入图像
        {
          'labels': [...],  # 整数列表，长度为 (N, )
          'scores': [...],  # 浮点列表，长度为 (N, )
          'bboxes': [...],  # 2d 列表，形状为 (N, 4)，格式为 [min_x, min_y, max_x, max_y]
        },
        ...
      ],
      'visualization' : [
        array(..., dtype=uint8),
      ]
  }
```

如果你想要从模型中获取原始输出，可以将 `return_datasamples` 设置为 `True` 来获取原始的 [DataSample](advanced_guides/structures.md)，它将存储在 `predictions` 中。

#### 储存结果

除了从返回值中获取预测结果，你还可以通过设置 `out_dir` 和 `no_save_pred`/`no_save_vis` 参数将预测结果和可视化结果导出到文件中。

```python
inferencer('demo/demo.jpg', out_dir='outputs/', no_save_pred=False)
```

结果目录结构如下：

```text
outputs
├── preds
│   └── demo.json
└── vis
    └── demo.jpg
```

#### 批量推理

你可以通过设置 `batch_size` 来自定义批量推理的批大小。默认批大小为 1。

### API

这里列出了推理器详尽的参数列表。

- **DetInferencer.\_\_init\_\_():**

| 参数            | 类型       | 默认值  | 描述                                                                                                                                                                                                                        |
| --------------- | ---------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`         | str , 可选 | None    | 配置文件的路径或 metafile 中定义的模型名称。例如，可以是 'rtmdet-s' 或 'rtmdet_s_8xb32-300e_coco' 或 'configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'。如果未指定模型，用户必须提供 MMEngine 保存的包含配置字符串的 "weights"。 |
| `weights`       | str, 可选  | None    | 模型权重文件的路径。如果未指定且 `model` 是 metafile 中的模型名称，权重将从 metafile 中加载。                                                                                                                               |
| `device`        | str, 可选  | None    | 推理使用的设备，接受 `torch.device` 允许的所有字符串。例如，'cuda:0' 或 'cpu'。如果为 None，将自动使用可用设备。 默认为 None。                                                                                              |
| `scope`         | str, 可选  | 'mmdet' | 模型的”域名“。                                                                                                                                                                                                              |
| `palette`       | str        | 'none'  | 用于可视化的配色。优先顺序为 palette -> config -> checkpoint。                                                                                                                                                              |
| `show_progress` | bool       | True    | 控制是否在推理过程中显示进度条。                                                                                                                                                                                            |

- **DetInferencer.\_\_call\_\_()**

| 参数                 | 类型                    | 默认值   | 描述                                                                                                                                                                                                                            |
| -------------------- | ----------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `inputs`             | str/list/tuple/np.array | **必需** | 它可以是一个图片/文件夹的路径，一个 numpy 数组，或者是一个包含图片路径或 numpy 数组的列表/元组                                                                                                                                  |
| `batch_size`         | int                     | 1        | 推理的批大小。                                                                                                                                                                                                                  |
| `return_vis`         | bool                    | False    | 是否返回可视化结果。                                                                                                                                                                                                            |
| `show`               | bool                    | False    | 是否在弹出窗口中显示可视化结果。                                                                                                                                                                                                |
| `wait_time`          | float                   | 0        | 弹窗展示可视化结果的时间间隔。                                                                                                                                                                                                  |
| `no_save_vis`        | bool                    | False    | 是否将可视化结果保存到 `out_dir`。默认为保存。                                                                                                                                                                                  |
| `draw_pred`          | bool                    | True     | 是否绘制预测的边界框。                                                                                                                                                                                                          |
| `pred_score_thr`     | float                   | 0.3      | 显示预测框的最低置信度。                                                                                                                                                                                                        |
| `return_datasamples` | bool                    | False    | 是否将结果作为 `DetDataSample` 返回。 如果为 False，则结果将被打包到一个 dict 中。                                                                                                                                              |
| `print_result`       | bool                    | False    | 是否将推理结果打印到控制台。                                                                                                                                                                                                    |
| `no_save_pred`       | bool                    | True     | 是否将推理结果保存到 `out_dir`。默认为不保存。                                                                                                                                                                                  |
| `out_dir`            | str                     | ''       | 结果的输出目录。                                                                                                                                                                                                                |
| `texts`              | str/list\[str\]，可选   | None     | 文本提示词。                                                                                                                                                                                                                    |
| `stuff_texts`        | str/list\[str\]，可选   | None     | 物体文本提示词。                                                                                                                                                                                                                |
| `custom_entities`    | bool                    | False    | 是否使用自定义实体。只用于 GLIP 算法。                                                                                                                                                                                          |
| \*\*kwargs           |                         |          | 传递给 :meth:`preprocess`、:meth:`forward`、:meth:`visualize` 和 :meth:`postprocess` 的其他关键字参数。kwargs 中的每个关键字都应在相应的 `preprocess_kwargs`、`forward_kwargs`、`visualize_kwargs` 和 `postprocess_kwargs` 中。 |

## 演示脚本样例

我们还提供了四个演示脚本，它们是使用高层编程接口实现的。[源码在此](https://github.com/open-mmlab/mmdetection/blob/main/demo) 。

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

#### 视频样例，显卡加速版本

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

### 大图推理样例

这是在大图上进行切片推理的脚本。

```shell
python demo/large_image_demo.py \
	${IMG_PATH} \
	${CONFIG_FILE} \
	${CHECKPOINT_FILE} \
	--device ${GPU_ID}  \
	--show \
	--tta  \
	--score-thr ${SCORE_THR} \
	--patch-size ${PATCH_SIZE} \
	--patch-overlap-ratio ${PATCH_OVERLAP_RATIO} \
	--merge-iou-thr ${MERGE_IOU_THR} \
	--merge-nms-type ${MERGE_NMS_TYPE} \
	--batch-size ${BATCH_SIZE} \
	--debug \
	--save-patch
```

运行样例:

```shell
# inferecnce without tta
wget -P checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth

python demo/large_image_demo.py \
    demo/large_image.jpg \
    configs/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py \
    checkpoint/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth

# inference with tta
wget -P checkpoint https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth

python demo/large_image_demo.py \
    demo/large_image.jpg \
    configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    checkpoint/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --tta
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

如果想进行多种类型的识别，需要使用 `xx. xx` 的格式在 `--texts` 字段后声明目标类型:

```shell
python demo/image_demo.py demo/demo.jpg glip_tiny_a_mmdet-b3654169.pth --texts 'bench. car'
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
