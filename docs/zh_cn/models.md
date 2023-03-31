# 模型

在深度学习任务中，我们通常将神经网络定义为一个模型，并且这个模型是算法的核心。[MMEngine](https://github.com/open-mmlab/mmengine)抽象了一个统一的模型[BaseModel](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model/base_model.py#L16)来作为标准化训练、测试和其他过程的接口。MMSegmentation中实现的所有模型都继承自`BaseModel`，在MMSegmentation中我们实现了前向传播并添加了一些语义分割算法的函数。

## 常见组件

### Segmentor

在MMSegmentation中，我们将网络架构抽象为一个**Segmentor**，它是一个包含所有网络组件的模型。我们已经实现了**EncoderDecoder**和**CascadeEncoderDecoder**，它们通常由**数据预处理器**、**骨干网络**、**解码头**和**辅助头**组成。

### 数据预处理器(Data preprocessor)

**数据预处理器** 是将数据复制到目标设备并将数据预处理为模型输入格式的部分。

### 骨干网络(Backbone)

**骨干网络** 是将图像转换为特征映射的部分，例如一个没有最后全一个全连接层的**ResNet-50**。

### Neck

**Neck**是连接骨干网络和头部的部分。它会对骨干网络生成的原始特征映射进行一些细化或重新配置。例如**Feature Pyramid Network (FPN)**。

### 解码头(Decode head)

**解码头** 是将特征映射转换为分割掩模的部分，例如**PSPNet**。

### 辅助头(Auxiliary head)

**辅助头**是可选组件，将特征映射转换为仅用于计算辅助损失的分割掩模。

## 基本接口

MMSegmentation 封装了 `BaseModel` 并实现了 [BaseSegmentor](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/models/segmentors/base.py#L15) 类，它主要提供了 `forward`, `train_step`, `val_step` 和 `test_step` 这些接口。下面将详细介绍这些接口。

### 前向传播

<center>
  <img src='../../../resources/encoder_decoder_dataflow.png' />
  <center>EncoderDecoder dataflow</center>
</center>

<center>
  <center><img src='../../../resources/cascade_encoder_decoder_dataflow.png' /></center>
  <center>CascadeEncoderDecoder dataflow</center>
</center>

`forward`方法返回训练、验证、测试以及简单推理过程的损失或预测。

该方法应接受三种模式：“tensor”、“predict”和“loss”：

- “tensor”：前向传播整个网络并返回张量或张量元组而没有任何后处理，就像普通的 `nn.Module` 一样。
- “predict”：前向传播并返回预测结果，这些结果已完全处理为一系列`SegDataSample`列表。
- “loss”：前向传播并根据给定的输入和数据样本返回一个损失的 `dict`。

**注意：** [SegDataSample](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/structures/seg_data_sample.py) 是 MMSegmentation 的数据结构接口，它用作不同组件之间的接口。`SegDataSample` 实现了抽象数据元素 `mmengine.structures.BaseDataElement`，请参阅 [SegDataSample文档](https://mmsegmentation.readthedocs.io/en/1.x/advanced_guides/structures.html) 和 [MMEngine](https://github.com/open-mmlab/mmengine) 中的[数据元素文档](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html)以获取更多信息。

请注意，本方法不处理反向传播或优化器更新，这些操作在 `train_step` 方法中完成。

参数：

- inputs (torch.Tensor) - 一般情况下是形状为 (N, C, ...) 的输入张量。
- data_sample (list\[[SegDataSample](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/structures/seg_data_sample.py)\]) - 语义数据样本。通常包括诸如`metainfo`和`gt_sem_seg`之类的信息。默认为None。
- mode (str) - 返回什么类型的值。默认为“tensor”。


返回：

  - 如果 `mode == "loss"`，返回一个用于反向传播和日志记录的损失张量字典 `dict`。
  - 如果 `mode == "predict"`，返回 `SegDataSample` 的 `list`，推理结果将逐渐添加到传递给前向方法的 `data_sample` 参数中，每个 `SegDataSample` 包含以下键：
    - pred_sem_seg (`PixelData`): 语义分割的预测。
    - seg_logits (`PixelData`): 正规化之前的语义分割的预测对应的语义分割的"logits"。
  - 如果 `mode == "tensor"`，返回一个用于自定义使用的`tensor`、`tuple of tensor`或张量字典(`dict` of `tensor`)。

### 预测模式

我们在[配置文档](../user_guides/1_config.md)中简要描述了模型配置的字段，这里我们详细说明`model.test_cfg`字段。`model.test_cfg`用于控制前向行为，`"predict"`模式下的`forward`方法可以在两种模式下运行：

- `whole_inference`：如果`cfg.model.test_cfg.mode == 'whole'`，模型将使用完整图像进行推理。

  一个`whole_inference`模式的示例配置：

  ```python
  model = dict(
    type='EncoderDecoder'
    ...
    test_cfg=dict(mode='whole')
  )
  ```

- `slide_inference`：如果`cfg.model.test_cfg.mode == 'slide'`，模型将通过滑动窗口进行推理。**注意**:如果选择`slide`模式，还应指定`cfg.model.test_cfg.stride`和`cfg.model.test_cfg.crop_size`。
  一个`slide_inference`模式的示例配置：

  ```python
  model = dict(
    type='EncoderDecoder'
    ...
    test_cfg=dict(mode='slide', crop_size=256, stride=170)
  )
  ```

### train_step

`train_step`方法调用`loss`模式的前向接口以获取损失`dict`。`BaseModel`类实现了包括预处理、模型前向传播、损失计算、优化和反向传播在内的默认模型训练过程。

参数：

- data（字典或元组或列表） - 从数据集中采样的数据。在MMSegmentation中，数据字典包含`inputs`和`data_samples`两个字段。
- optim_wrapper（OptimWrapper） - OptimWrapper实例用于更新模型参数。

**注意：** [OptimWrapper](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py#L17)提供了一个通用接口来更新参数，请参阅[MMEngine](https://github.com/open-mmlab/mmengine)中的优化器包装器[文档](https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.html)以获取更多信息。

返回值：

- Dict\[str, `torch.Tensor`\]: 用于记录的张量字典(`dict`).

<center>
  <img src='../../../resources/train_step.png' />
  <center>train_step dataflow</center>
</center>

### val_step

`val_step`方法调用`predict`模式的前向接口并返回预测结果，该结果进一步传递给评估器的处理接口和钩子的`after_val_iter`接口。

参数：

- data（`dict`或`tuple`或`list`） - 从数据集中采样的数据。在MMSegmentation中，数据字典包含`inputs`和`data_samples`两个字段。

返回值：

- `list` - 给定数据的预测。

<center>
  <img src='../../../resources/test_step.png' />
  <center>test_step/val_step dataflow</center>
</center>

### test_step

`BaseModel`以与`val_step`相同的方式实现了`test_step`

## 数据预处理器

MMSegmentation实现的[SegDataPreProcessor](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/models/data_preprocessor.py#L13)继承自[MMEngine](https://github.com/open-mmlab/mmengine)实现的[BaseDataPreprocessor](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model/data_preprocessor.py#L18)，并提供了数据预处理和将数据复制到目标设备的功能。

在构建阶段，运行器将模型放到指定设备，而数据由`train_step`, `val_step`和`test_step`中的[SegDataPreProcessor](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/models/data_preprocessor.py#L13)携带到指定设备，处理后的数据进一步传递给模型。

`SegDataPreProcessor`构造函数的参数：

- mean (Sequence\[Number\], 可选) - R、G、B通道的像素平均值。默认为None。
- std (Sequence\[Number\], 可选) - R、G、B通道的像素标准差。默认为None。
- size (tuple, 可选) - 固定填充大小。
- size_divisor (int, 可选) - 填充大小的除数。
- pad_val (float, 可选) - 填充值。默认为0。
- seg_pad_val (float, 可选) - 分割图的填充值。默认为255。
- bgr_to_rgb (bool) - 是否将图像从BGR转换为RGB。默认为False。
- rgb_to_bgr (bool) - 是否将图像从RGB转换为BGR。默认为False。
- batch_augments (list\[dict\], 可选) - batch级数据增强。默认为None。

数据将按如下方式处理：

- 整理并将数据移动到目标设备。
- 使用定义的`pad_val`填充输入大小，使用定义的`seg_pad_val`填充分割图。
- 将输入堆叠到batch_inputs。
- 如果输入的形状为（3，H，W），则将输入从bgr转换为rgb。
- 使用定义的std和mean对图像进行标准化。
- 在训练期间进行批量数据增强，如Mixup和Cutmix。

`forward`方法的参数：

- data（dict） - 从dataloader采样的数据。
- training (bool) - 是否训练时启用数据增强。

`forward` 方法的返回：

- Dict: 与模型输入相同格式的数据。
