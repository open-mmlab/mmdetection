# MMDetection v2.x 兼容性说明

## MMDetection v2.14.0

### MMCV 版本

为了修复 EvalHook 优先级过低的问题，MMCV v1.3.8 中所有 hook 的优先级都重新进行了调整，因此 MMDetection v2.14.0 需要依赖最新的 MMCV v1.3.8 版本。 相关信息请参考[PR #1120](https://github.com/open-mmlab/mmcv/pull/1120) ，相关问题请参考[#5343](https://github.com/open-mmlab/mmdetection/issues/5343) 。

### SSD 兼容性

在 v2.14.0 中，为了使 SSD 能够被更灵活地使用，[PR #5291](https://github.com/open-mmlab/mmdetection/pull/5291) 重构了 SSD 的 backbone、neck 和 head。用户可以使用 tools/model_converters/upgrade_ssd_version.py 转换旧版本训练的模型。


```shell
python tools/model_converters/upgrade_ssd_version.py ${OLD_MODEL_PATH} ${NEW_MODEL_PATH}

```

- OLD_MODEL_PATH：旧版 SSD 模型的路径。
- NEW_MODEL_PATH：保存转换后模型权重的路径。

## MMDetection v2.12.0

在 v2.12.0 到 v2.18.0（或以上）版本的这段时间，为了提升通用性和便捷性，MMDetection 正在进行大规模重构。在升级到 v2.12.0 后 MMDetection 不可避免地带来了一些 BC Breaking，包括 MMCV 的版本依赖、模型初始化方式、模型 registry 和 mask AP 的评估。

### MMCV 版本

MMDetection v2.12.0 依赖 MMCV v1.3.3 中新增加的功能，包括：使用 `BaseModule` 统一参数初始化，模型 registry，以及[Deformable DETR](https://arxiv.org/abs/2010.04159) 中的 `MultiScaleDeformableAttn` CUDA 算子。
注意，尽管 MMCV v1.3.2 已经包含了 MMDet 所需的功能，但是存在一些已知的问题。我们建议用户跳过 MMCV v1.3.2 使用 v1.3.3 版本。

### 统一模型初始化

为了统一 OpenMMLab 项目中的参数初始化方式，MMCV 新增加了 `BaseModule` 类，使用 `init_cfg` 参数对模块进行统一且灵活的初始化配置管理。
现在用户需要在训练脚本中显式调用 `model.init_weights()` 来初始化模型（例如 [这行代码](https://github.com/open-mmlab/mmdetection/blob/master/tools/train.py#L162) ，在这之前则是在 detector 中进行处理的。
**下游项目必须相应地更新模型初始化方式才能使用 MMDetection v2.12.0**。请参阅 [PR #4750](https://github.com/open-mmlab/mmdetection/pull/4750) 了解详情。

### 统一模型 registry

为了能够使用在其他 OpenMMLab 项目中实现的 backbone，MMDetection v2.12.0 继承了在 MMCV (#760) 中创建的模型 registry。
这样，只要 OpenMMLab 项目实现了某个 backbone，并且该项目也使用 MMCV 中的 registry，那么用户只需修改配置即可在 MMDetection 中使用该 backbone，不再需要将代码复制到 MMDetection 中。 更多详细信息，请参阅 [PR #5059](https://github.com/open-mmlab/mmdetection/pull/5059) 。

### Mask AP 评估

在 [PR #4898](https://github.com/open-mmlab/mmdetection/pull/4898) 和 v2.12.0 之前，对小、中、大目标的 mask AP 的评估是基于其边界框区域而不是真正的 mask 区域。
这导致 `APs` 和 `APm` 变得更高但 `APl` 变得更低，但是不会影响整体的 mask AP。 [PR #4898](https://github.com/open-mmlab/mmdetection/pull/4898) 删除了 mask AP 计算中的 `bbox` ，改为使用 mask 区域。
新的计算方式不会影响整体的 mask AP 评估，与 [Detectron2](https://github.com/facebookresearch/detectron2/)一致。

## 与 MMDetection v1.x 的兼容性

MMDetection v2.0 经过了大规模重构并解决了许多遗留问题。 MMDetection v2.0 不兼容 v1.x 版本，在这两个版本中使用相同的模型权重运行推理会产生不同的结果。 因此，MMDetection v2.0 重新对所有模型进行了 benchmark，并在 model zoo 中提供了新模型的权重和训练记录。

新旧版本的主要的区别有四方面：坐标系、代码库约定、训练超参和模块设计。

### 坐标系

新坐标系与 [Detectron2](https://github.com/facebookresearch/detectron2/) 一致，
将最左上角的像素的中心视为坐标原点 (0, 0) 而不是最左上角像素的左上角。 因此 COCO 边界框和分割标注中的坐标被解析为范围 `[0，width]` 和 `[0，height]` 中的坐标。 这个修改影响了所有与 bbox 及像素选择相关的计算，变得更加自然且更加准确。

- 在新坐标系中，左上角和右下角为 (x1, y1) (x2, y2) 的框的宽度及高度计算公式为 `width = x2 - x1` 和 `height = y2 - y1`。
  在 MMDetection v1.x 和之前的版本中，高度和宽度都多了 `+ 1` 的操作。
  本次修改包括三部分：

  1. box 回归中的检测框变换以及编码/解码。
  2. IoU 计算。这会影响 ground truth 和检测框之间的匹配以及 NMS 。但对兼容性的影响可以忽略不计。
  3. Box 的角点坐标为浮点型，不再取整。这能使得检测结果更为准确，也使得检测框和 RoI 的最小尺寸不再为 1，但影响很小。

- Anchor 的中心与特征图的网格点对齐，类型变为 float。
  在 MMDetection v1.x 和之前的版本中，anchors 是 `int` 类型且没有居中对齐。
  这会影响 RPN 中的 Anchor 生成和所有基于 Anchor 的方法。

- ROIAlign 更好地与图像坐标系对齐。新的实现来自 [Detectron2](https://github.com/facebookresearch/detectron2/tree/master/detectron2/layers/csrc/ROIAlign) 。
  当 RoI 用于提取 RoI 特征时，与 MMDetection v1.x 相比默认情况下相差半个像素。
  能够通过设置 `aligned=False` 而不是 `aligned=True` 来维持旧版本的设置。

- Mask 的裁剪和粘贴更准确。

  1. 我们使用新的 RoIAlign 来提取 mask 目标。 在 MMDetection v1.x 中，bounding box 在提取 mask 目标之前被取整，裁剪过程是 numpy 实现的。 而在新版本中，裁剪的边界框不经过取整直接输入 RoIAlign。 此实现大大加快了训练速度（每次迭代约加速 0.1 秒，1x schedule 训练 Mask R50 时加速约 2 小时）并且理论上会更准确。
  2. 在 MMDetection v2.0 中，修改后的 `paste_mask()` 函数应该比之前版本更准确。 此更改参考了 [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/masks.py) 中的修改，可以将 COCO 上的 mask AP 提高约 0.5%。

### 代码库约定

- MMDetection v2.0 更改了类别标签的顺序，减少了回归和 mask 分支里的无用参数并使得顺序更加自然（没有 +1 和 -1）。
  这会影响模型的所有分类层，使其输出的类别标签顺序发生改变。回归分支和 mask head 的最后一层不再为 K 个类别保留 K+1 个通道，类别顺序与分类分支一致。

  - 在 MMDetection v2.0 中，标签 “K” 表示背景，标签 [0, K-1] 对应于 K = num_categories 个对象类别。

  - 在 MMDetection v1.x 及之前的版本中，标签 “0” 表示背景，标签 [1, K] 对应 K 个类别。

  - **注意**：softmax RPN 的类顺序在 version<=2.4.0 中仍然和 1.x 中的一样，而 sigmoid RPN 不受影响。从 MMDetection v2.5.0 开始，所有 head 中的类顺序是统一的。

- 不使用 R-CNN 中的低质量匹配。在 MMDetection v1.x 和之前的版本中，`max_iou_assigner` 会在 RPN 和 R-CNN 训练时给每个 ground truth 匹配低质量框。我们发现这会导致最佳的 GT 框不会被分配给某些边界框，
  因此，在MMDetection v2.0 的 R-CNN 训练中默认不允许低质量匹配。这有时可能会稍微改善 box AP（约为 0.1%）。

- 单独的宽高比例系数。在 MMDetection v1.x 和以前的版本中，`keep_ratio=True` 时比例系数是单个浮点数，这并不准确，因为宽度和高度的比例系数会有一定的差异。 MMDetection v2.0 对宽度和高度使用单独的比例系数，对 AP 的提升约为 0.1%。

- 修改了 config 文件名称的规范。 由于 model zoo 中模型不断增多， MMDetection v2.0 采用新的命名规则：

  ```shell
  [model]_(model setting)_[backbone]_[neck]_(norm setting)_(misc)_(gpu x batch)_[schedule]_[dataset].py
  ```

  其中 (`misc`) 包括 DCN 和 GCBlock 等。更多详细信息在 [配置文件说明文档](config.md) 中说明

- MMDetection v2.0 使用新的 ResNet Caffe backbone 来减少加载预训练模型时的警告。新 backbone 中的大部分权重与以前的相同，但没有 `conv.bias`，且它们使用不同的 `img_norm_cfg`。因此，新的 backbone 不会报 `unexpected keys` 的警告。

### 训练超参

训练超参的调整不会影响模型的兼容性，但会略微提高性能。主要有：

- 通过设置 `nms_post=1000` 和 `max_num=1000`，将 nms 之后的 proposal 数量从 2000 更改为 1000。使 mask AP 和 bbox AP 提高了约 0.2%。

- Mask R-CNN、Faster R-CNN 和 RetinaNet 的默认回归损失从 smooth L1 损失更改为 L1 损失，使得 box AP 整体上都有所提升（约 0.6%）。但是，将 L1-loss 用在 Cascade R-CNN 和 HTC 等其他方法上并不能提高性能，因此我们保留这些方法的原始设置。

- 为简单起见，RoIAlign 层的 `sampling_ratio` 设置为 0。略微提升了 AP（约 0.2% 绝对值）。

- 为了提升训练速度，默认设置在训练过程中不再使用梯度裁剪。大多数模型的性能不会受到影响。对于某些模型（例如 RepPoints），我们依旧使用梯度裁剪来稳定训练过程从而获得更好的性能。

- 因为不再默认使用梯度裁剪，默认 warmup 比率从 1/3 更改为 0.001，以使模型训练预热更加平缓。不过我们重新进行基准测试时发现这种影响可以忽略不计。

### 将模型从 v1.x 升级至 v2.0

用户可以使用脚本 `tools/model_converters/upgrade_model_version.py` 来将 MMDetection 1.x 训练的模型转换为 MMDetection v2.0。转换后的模型可以在 MMDetection v2.0 中运行，但性能略有下降（小于 1% AP）。
详细信息可以在 `configs/legacy` 中找到。

## pycocotools 兼容性

`mmpycocotools` 是 OpenMMLab 维护的 `pycocotools` 的复刻版，适用于 MMDetection 和 Detectron2。
在 [PR #4939](https://github.com/open-mmlab/mmdetection/pull/4939) 之前，由于 `pycocotools` 和 `mmpycocotool` 具有相同的包名，如果用户已经安装了 `pyccocotools`（在相同环境下先安装了 Detectron2 ），那么 MMDetection 的安装过程会跳过安装 `mmpycocotool`。 导致 MMDetection 缺少 `mmpycocotools` 而报错。
但如果在 Detectron2 之前安装 MMDetection，则可以在相同的环境下工作。
[PR #4939](https://github.com/open-mmlab/mmdetection/pull/4939) 弃用 mmpycocotools，使用官方 pycocotools。
在 [PR #4939](https://github.com/open-mmlab/mmdetection/pull/4939) 之后，用户能够在相同环境下安装 MMDetection 和 Detectron2，不再需要关注安装顺序。
