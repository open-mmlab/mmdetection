# Detection Transformer

Detection Transformers (DETRs) 是一系列基于 Transformer 的目标检测算法。

多数的DETR算法的处理流程如下：先用 骨干网络（backbone）和 颈部网络（neck）提取图像特征。然后使用一个由 编码器（encoder）和 解码器（decoder）构成的 Transformer 网络处理特征。该 Transformer 网络直接并行输出一组 查询（queries）。每个查询对应着一个预测，这个预测可能是目标，也可能是 `no object` 即“非目标”类。

![DETR_overall](.\DETR_overall.png)

在 MMDetection 中实现了全新的 DETRs，其主要有以下四项特征：

#### 1. 统一的 DETRs 检测器基类

MMDetection 根据现有的 DETRs 算法，设计了统一的检测器基类 `DetectionTransformer`。在该基类中标准化了 DETRs 的整体结构和前向过程。文件位置为 mmdet/models/detectors/base_detr.py。

每个 DETR 模型都继承自 `DetectionTransformer`，主要由六个组件（components）组成：骨干网络 `backbone`，颈部网络 `neck`，位置编码模块 `positional_encoding`，编码器 `encoder`，解码器 `decoder`，和 检测头 `bbox_head`。其中：骨干网络 和 颈部网络 主要用于提取输入图像的特征；位置编码模块用于编码每个特征点的位置；编码器用于处理颈部网络输出的特征；解码器用于从编码器输出的特征中提取目标特征，获得目标查询；检测头用于在解码器输出的查询上获得预测和计算损失。

前向过程主要为 提取特征，Transformer处理特征，和 用检测头获得预测和计算损失 三部分。经过设计，Transformer 的前向过程又分为 `pre_transformer`, `forward_encoder`, `pre_decoder`, and `forward_decoder` 四个部分，每个部分对应一个检测器类的私有函数。这些函数之间的参数流总结如下：

![DETR_forward_process](.\DETR_forward_process.png)

对应的代码如下：

```Python
def forward_transformer(self, img_feats: Tuple[Tensor],
                        batch_data_samples: OptSampleList = None) -> Dict:

    encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
        img_feats, batch_data_samples)

    encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

    tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
    decoder_inputs_dict.update(tmp_dec_in)

    decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
    head_inputs_dict.update(decoder_outputs_dict)

    return head_inputs_dict
```

#### 2. 更加简洁的部件

新版 MMDetection 中，多数部件类的实现变得更加简洁了，如 `DetrTransformerEncoderLayer`, `DetrTransformerDecoder`, 和 `DetrHead`。

多数代码库中的 Transformer 类 `XTransformer` 对象 被去除了，检测器会直接初始化 编码器 `XTransformerEncoder` 对象 和 解码器 `XTransformerDecoder` 对象。

为了兼容多种使用情况，旧版 MMDetection 中的 Transformer 模块，包括 编码器类`XTransformerEncoder`, 编码器层类 `XTransformerEncoderLayer`, 解码器类 `XTransdormerDecoder`, 解码器层类 `XTransformerDecoderLayer`，设计较为复杂，可读性较低，有着过度兼容的问题。新版 MMDetection 中，这些模块使用更简单直白的方式实现，可读性更高，逻辑更清晰。如下例：

```Python
class DetrTransformerEncoder(BaseModule):
    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        self.layers = ModuleList([
            DetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)])
        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs) -> Tensor:
        for layer in self.layers:
            query = layer(query, query_pos, key_padding_mask, **kwargs)
        return query
```

此外，检测头类 `XHead` 变得更加轻量化。新版 MMDetection 将 Transformer 的部件从检测头对象内移到检测器对象内，这使检测头得以专注于获得检测结构和计算损失的任务中。

#### 3. 可读性更高、更好用的代码

新版 MMDetection 重构了所有 DETRs 算法的代码，新的代码可读性更高，更好用。

重构后的各模块类具有更加合理的的设计和统一的实现风格，这使得阅读 MMDetection 支持的 DETRs 或实现自定义的DETR都变得十分简单。用户可以参考 [自定义一个DETR检测器](<>) 来实现一个新的 DETR。

旧版 MMDetection 在构建模块时过度使用了注册机制，导致用户体验较差。新版 MMDetection 直接初始化这些模块，代码更加直观，并且支持代码跳转。

```python
# Original implementation
self.encoder = build_transformer_layer_sequence(encoder)
self.decoder = build_transformer_layer_sequence(decoder)
# Refactored implementation
self.encoder = DetrTransformerEncoder(**self.encoder)
self.decoder = DetrTransformerDecoder(**self.decoder)
# Show the module class directly, support code jumping.
```

新版 MMDetection 的模块都附带更加详细的文档和注释，来帮助读者阅读和理解算法代码。

#### 4. 更多 SOTA 的 DETR 算法

在最新的 MMDetection 中，原本支持的 DETRs 被全部重构，又新增了几个 SOTA 的 DETR 算法。

目前支持的 DETRs：

- [x] [DETR](https://arxiv.org/abs/2005.12872) (ECCV'2020)
- [x] [Deformable-DETR](https://arxiv.org/abs/2010.04159v4) (ICLR'2021)
- [x] [Conditional-DETR](https://arxiv.org/abs/2108.06152v2) (ICCV'2021)
- [x] [DAB-DETR](https://arxiv.org/abs/2201.12329) (ICLR'2022)
- [x] [DINO](https://arxiv.org/abs/2203.03605) (ArXiv'2022)

### 基本概念

#### 图像特征 和 序列特征

Transformer 是自然语言处理领域的主流模型，其处理的数据通常为序列（Sequence）。序列数据一般有三个维度：批大小 `B`（batch size）、序列长度/ `N`（length of sequence）、通道数/嵌入维度 `C`（channel / embedding dimension）。该批数据中的每个样本都可以理解为 `N` 个 维度为 `C` 的向量。序列的特征也是相同的表示形式。

在计算机视觉领域，图像数据一般有四个维度：批大小 `B` （batch size）、通道数 `C` （channel）、图高 `H`（height）、图宽 `W`（width）。图像的特征也是相同的表达形式。

因此，骨干网络和颈部网络提取的图像特征 `(B, C, H, W)` 在输入 Transformer 之前，要转化为序列特征 `(B, N, C)` 的形式。通常先展平（flatten）宽高两维，再进行维度替换（permute），获得的序列的 `N` 即为 `H x W`。如果需要将序列特征还原成图像特征，只需要进行上述操作的逆运算，不需要额外的参数。

图像特征转化为序列特征的逻辑通常在各检测器的 `pre_transformer` 中实现，采用的方式和多数代码中稍有不同，该方式支持动态导出到 ONNX：

```python
# [bs, c, h, w] -> [bs, h*w, c]
# Most codebase:
feat = feat.flatten(2).permute(0, 2, 1)
# MMDetection:
feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
```

展平宽高后，特征丢失了空间位置信息。因此在 DETRs 中通常采用2D的位置编码，对每个特征点在特征图的行列位置进行编码，作为该特征点的位置嵌入。更多细节可以参考 [DETR 的位置嵌入](<>)。

上述操作能支持单尺度的特征图转换为序列特征，进而被 Transformer 处理。而对于多尺度特征图，通常需要记录更多的信息。

多尺度特征图通常为多个图像特征的元组，第`l`个层级（level）的图像特征形式为 `(B, C, H_l, W_l)`，其中`H_l`, `W_l` 为该特征图的高宽。将该元组转化为序列特征的方式通常先对每个特征图进行展平和维度替换的操作，再将获得的几个序列在 `N` 这一维度合并（concat）起来：

```Python
feat_flatten = []
for lvl, feat in enumerate(mlvl_feats):
    batch_size, c, h, w = feat.shape
    # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
    feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
    feat_flatten.append(feat)
feat_flatten = torch.cat(feat_flatten, 1)
```

多尺度特征图的位置嵌入中通常增加特征层级嵌入（level embeddings），来分辨特征图的层级。特征层级嵌入通常和位置嵌入直接相加，统一用 `lvl_pos_embed` 来表示。

此外，为了支持对多尺度特征图的更多特殊操作，通常需要记录一些额外的信息。例如 每个尺度的特征空间大小 `spatial shape`; 由 `spatial shape` 可以获得每个特征图在 `N` 这一维度的起始索引 `lvl_start_index`。通过这两个参数可以反过来将 `(B, N, C)` 格式的序列特征还原成 `(B, C, H_l, W_l)` 格式的多尺度特征；也可以支持多尺度特征交互操作，例如可变形注意力（Deformable Attention）。

#### DETR 的位置嵌入

Transformer 的自注意力的计算过程通常是先用查询（query）和键（key）生成注意力，将注意力作为权重将值（value）加权。由于该计算过程具有置换不变性（permutation invariance），多数自注意力在计算前通常对查询、键和值嵌入位置编码。

在 DETR 系列算法的 Transformer 的注意力模块的输入中，也进行了位置嵌入。与多数情况不同的是，DETR 只对查询和键进行位置嵌入。

.........

### 约定

#### 参数名

在各种 DETR 的代码中，query 这一变量名常常具有多重含义，导致用户很容易混淆变量的意义和作用。在新版 MMDetection 中，我们对参数名进行约定，统一实现各算法。

在 DETR 相关模块中，共采用了两种主要的命名规则，对应两个层面：

在检测器的层面：骨干网路和颈部网络提取的特征图称为 `img_feats`。`pre_transformer` 将特征图转换成输入编码器的特征序列，称为 `feat`。特征序列对应的掩码和位置编码分别称为 `feat_mask` 和 `feat_pos`。编码器的输出被称为 `memory`，对应的掩码称为 `memory_mask`。解码器中的查询为 `query`，在多数文章中被称为 "content query"，对应的位置嵌入称为 `query_pos`，在多数的文章中被称为 "positional query", "spatial query", 或 "object query"。我们用下图概括这些参数的命名规则：

在 Transformer 部件模块、注意力模块等深层模块的层面：按照注意力的输入参数名命名，即 查询为 `query`、键为 `key`, 且 值为 `value`，查询和健对应的位置分别为 `query_pos` 和 `key_pos`。由于输入图像在批处理时会采用零填充的方式对齐图像尺寸，在计算时将记录填充位置的掩码命名为 `key_padding_mask`。在解码器中也可以指定 `self_attn_mask` 和 `cross_attn_mask` 作为不同注意力模块的掩码。

检测器类的 `forward_encoder` 和 `forward_decoder` 方法的本质作用是 对两种参数命名规则进行映射。

因此读者在阅读和使用代码时要注意判断是哪种命名规则。尤其对于 `query` 这个参数，如果是检测器中的参数，应当特指的是解码器的查询，如果是 Transformer 部件的参数，应当代表的是注意力计算中的查询。

#### 统一的参数流

### 自定义一个 DETR 检测器

#### 实现 Transformer 的 组件

#### 实现检测器类

#### 实现检测头类

#### 示例: 实现 DINO
