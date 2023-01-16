# Detection Transformer

Detection Transformers (DETRs) are a series of Transformer-based object detectors.

In most DETRs, the features extracted from backbone and neck were fed into a Transformer which is composed of an encoder and a decoder. The Transformer directly outputs a set of queries in parallel. Each query corresponds to one prediction, which may be an object or `no object`.

![DETR_overall](.\DETR_overall.png)

There is a brand new implementation of DETRs in MMDetection, whose features are fourfold:

#### 1. A unified base detector class for DETRs.

According to the existing DETRs, a unified base detector class [`DetectionTransformer`](../../../mmdet/models/detectors/base_detr.py) is designed to standardize the architectures and forward procedures of DETRs.

Each detector inherited from `DetectionTransformer` is mainly consists of six components: `backbone`, `neck`, `positional_encoding`, `encoder`, `decoder`, and `bbox_head`. The `backbone` and `neck` extract features of input batch. The `positional_encoding` encodes the position of each feature points. The `encoder` processes output features of neck. The `decoder` pools encoder features of objects into the queries. At last, the `bbox_head` makes predictions and calculates losses on output queries of decoder.

For the forward process, except for the earliest `extracting features` and the latest `calculating with head`, the intermediate forward process of transformer are designed as four steps: `pre_transformer`, `forward_encoder`, `pre_decoder`, and `forward_decoder`. The parameters flow among the functions are summarized as follow:

![DETR_forward_process](.\DETR_forward_process.png)

The corresponding implementation is as follow:

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

#### 2. More concise components.

The implementations of most components, such as `DetrTransformerEncoderLayer`, `DetrTransformerDecoder`, and `DetrHead`, become more concise.

The `XTransformer` class has been dropped. Instead, the detector initialize `XTransformerEncoder` and `XTransformerDecoder` directly.

The original intricate modules of Transformer, including `XTransformerEncoder`, `XTransformerEncoderLayer`, `XTransdormerDecoder`, `XTransformerDecoderLayer`, are replaced with straightforward modules which attend to clear logic rather than excessive compatibility. For example:

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

Additionally, the `XHead` becomes lightweight, because the Transformer components are moved to detector, which makes the head focus on getting detection results and calculating losses.

#### 3. Code with better readability and usability

The code of all DETR-related modules are refactored to enhance the readability and usability.

The refactored modules have reasonable designs and a uniform implementation style, which makes it easy to read the code of supported DETRs or implement a custom DETR. Users can refer to [Customize a DETR](<>) to implement a new DETR.

The overused registration mechanism for building modules are replaced with direct initialization, which benefits code reading and jumping.

```python
# Original implementation
self.encoder = build_transformer_layer_sequence(encoder)
self.decoder = build_transformer_layer_sequence(decoder)
# Refactored implementation
self.encoder = DetrTransformerEncoder(**self.encoder)
self.decoder = DetrTransformerDecoder(**self.decoder)
# Show the module class directly, support code jumping.
```

At last, the new modules come with more detailed doc-strings and comments.

#### 4. More SOTA DETRs

In the latest MMDetection, the supported DETRs have been refactored. Besides, implementation and weight of several SOTA DETRs have been available.

Supported DETRs:

- [x] [DETR](https://arxiv.org/abs/2005.12872) (ECCV'2020)
- [x] [Deformable-DETR](https://arxiv.org/abs/2010.04159v4) (ICLR'2021)
- [x] [Conditional-DETR](https://arxiv.org/abs/2108.06152v2) (ICCV'2021)
- [x] [DAB-DETR](https://arxiv.org/abs/2201.12329) (ICLR'2022)
- [x] [DINO](https://arxiv.org/abs/2203.03605) (ArXiv'2022)

### Basic conceptions

#### Image feature and sequence feature

Transformer is the prevalent architecture for nature language processing, which usually process sequence data. There are usually three dimensions of sequence data: batch size `B`, number of tokens / length of sequence `N`, embedding dimension / number of channels `C`.

In the computer vision community, there are usually four dimensions of image data: batch size `B`, number of channels `C`, image height `H`, image width `W`.

Hence, the image features of `(B, C, H, W)` extracted by backbone and neck should be transformed into sequence feature of `(B, N, C)`.  Specifically, the two dimensions `H` and `W` are flattened into a new dimension `N = H x W`. Then the `N` and `C` dimensions are permuted.

The transformation logic is usually implemented in the `pre_transformer` function of detector class as follow:

```python
feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
```

After the flattening, the spatial position information is lost. Hence, in DETRs, the 2D positional encoding is used to encode the row and column positions of each feature point into positional embeddings. More details can be found in [Positional embedding in DETRs](<>).

The aforementioned operations support the transformation of a single-level feature. For transformation of multi-level features, more information should be recorded.

The multi-level features is actually a tuple of multiple image features. The format of the `l`-th level image feature is `(B, C, H_l, W_l)`, where `H_l` and `W_l` are height and width, respectively. The tuple of image features is also transformed into a sequence feature. Specifically, each image feature is transformed into a sequence feature of length `N_l = H_l x W_l` by flattening and permuting. Then, the obtained sequence features are concatenate on the dimension `N = N_1 + ... + N_L`.

```python
feat_flatten = []
for lvl, feat in enumerate(mlvl_feats):
    batch_size, c, h, w = feat.shape
    # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
    feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
    feat_flatten.append(feat)
feat_flatten = torch.cat(feat_flatten, 1)
```

When processing multi-level features, the level embeddings are usually added to the positional embeddings to distinguish the feature levels. The sum is represented with `lvl_pos_embed`.

To support more operations on multi-scale features, some extra information should be introduced. For example, the feature `spatial shape` on each level, the `lvl_start_index` (the start sequence indexes of each feature level), and so on. The `spatial shape` and `lvl_start_index` can be used to restore the sequence feature of `(B, N, C)` to the tuple of multi-scale features of `B, C, H_l, W_l`. They can also support special multi-scale feature interaction operations, such as Deformable Attention.

<img src="C:\Users\lqy\Desktop\doc_detr\DETR_mlvl_feats2seq.png" style="zoom:50%;" />

#### Positional embedding of DETRs

There are positional embeddings for the inputs of attention modules in DETRs. Unlike most cases, DETRs only embed for queries and keys, and not embed for values. Moreover, DETRs embed positions of both spatial directions, i.e. row and column, namely 2D position encoding.

![](C:\Users\lqy\Desktop\doc_detr\DETR_positional_encoding.png)

The left sub-figure illustrates the 2D position encoding process: The positional embeddings for queries or keys have same embedding dimension with the queries or keys. In the 2D position encoding of DETRs, the dimension of `C` is divided into two partitions of `C/2` uniformly. The former one is embedded for row position and the latter one is embedded for the column position.

The right sub-figure is excerpted from DAB-DETR paper, which illustrates the positional embeddings of DETRs. The queries `Q` and keys `K` are all composed of two partitions: Content queries / keys which attend to object feature content, and positional queries / keys which attend to positional information. The values `V` have not positional partitions. For encoder: Content queries, content keys, and values are all from image features.  Positional queries and positional keys are from the 2D position embeddings of image features. For decoder: Content keys, values, and positional keys remain. Content queries are from the outputs of last decoder layer or the initial decoder queries. Positional queries are positional embeddings of the decoder queries.

#### Object detection paradigm of set prediction

Most DETRs were set prediction-based detectors. They eliminated the complicated components required by many convenient detectors, such as non-maximum suppression, anchor generation. They obtains a set of predictions, and each prediction includes a category and a bounding box. The classification results include all object categories and a  `no object`  class.

In training: The Hungarian algorithm is used to assign a prediction to each ground truth target. Both classification losses and boxes losses are calculated for the predictions which have been assigned with certain ground truth. While for the predictions which have not been assigned with any ground truth. Their target labels are then assigned with  `no object`  class and only classification losses are calculated.

In inference: The final detection results can be obtained by directly remove the predictions whose classification results are  `no object`  and predictions whose classification confidences are lower than a presupposed threshold, without complicated post-processing.

### Appointment

#### Parameter names

In various codebase of DETRs, there are multiple meanings for the parameter name `query`, which make it easy for users to confuse about the variable. In the new MMDetection, we appoint the meaning of parameter names and standardize our implementation uniformly.

There are two levels of naming schemes in our implementation:

On the level of detector modules: The feature maps extracted by backbone and neck are  `img_feats`. The `pre_transformer()` converts the image feature into the input sequence feature of encoder, namely  `feat`. The masks and positional embeddings corresponding to the sequence feature are `feat_mask` and `feat_pos`. The outputs of the encoder are `memory`, whose corresponding masks are `memory_mask`. The decoder queries are `query`, which are called 'content query' in most papers. The positional embeddings of the decoder queries are `query_pos`, which are called 'spatial query',  'positional query', and 'object query' in most papers.

On the level of deep modules, including Transformer components and attention modules: For the attention modules, the queries, keys, and values are `query`, `key`, and `value`, respectively. The positional embeddings corresponding to queries and keys are `query_pos` and `key_pos`, respectively. For encoder modules and encoder layer modules, the input parameters are named according to the `self_attn`. For decoder modules and decoder layer modules, the input parameters are named according to the `cross_attn`. Since the input images are padded to align shapes when collating a batch, the padding positions are recorded in a mask, namely `key_padding_mask`. The `self_attn_mask` and `cross_attn_mask` can also be specified in the decoder as attention masks for the two attention modules.

The role of `forward_encoder()` and `forward_decoder()` functions of detector classes is to map the two levels of naming schemes.

Hence, while reading or using the codebase, the users should attend to judge which kind of naming scheme is used, especially for the parameter named `query`. (If it is in the detector, it represents the decoder queries. If it is in the Transformer components or attention modules, it indicates the queries of the attention calculation.)

#### Unified data flow

##### 1. Two kinds of data flow: ***batch first*** and ***sequnce first***
In the previous versions of DETR's implementation, data (image, feature etc.) in the transformer is organized as `[N, B, C]` format, where `N` denotes sequence length, `B` denotes batch size, and `C` denotes embedding dimension. We refer this format to `sequence_first`. Sequence first data flow is originated from the NLP community, hence the CV community inherited this format naturally when introducing transformer into its methodology.

On the other hand, the classic data flow of CV community in CNN-like models is `[B, ...]` instead, namely `batch_first`.

##### 2. Why we unify the data flow to batch first?

In older implementations of DETR, the data flow is converted from batch first to sequence first before the transformer encoder, and the process is done **reversely** after the transformer decoder.

The repeated conversion of data flow henders the **readability** and **usability** of code. Therefore, we unify the data flow to `[B, N, C]` in all DETRs' implementation, we believe the unified batch first data flow is more friendly to the researchers of CV community.

##### 3. Examples

We demonstrate the unified data flow from three aspects: Detector, head, attention.

### Customize a DETR

#### Implement Transformer components

There are four types of Transformer component modules: `XTransformerEncoder`，`XTransformerEncoderLayer`，`XTransformerDecoder`，`XTransformerDecoderLayer`. We put these Transformer component modules of the support DETRs at mmdet/models/layers/transformer/xxx_layers.py.

In most case, a new DETR may reuse some existing components. The users should analysis and decide which components to reuse and which modules should be re-written. A new module usually only requires to inherit from an existing module and build on it with minor modifications.

After inheriting an existing component module, the `_init_layers()` and `forward()` can be re-written according to the requirement. For `XTransformerEncoder._init_layers()` and `XTransformerDecoder._init_layers()`, the attributes `self.layers` and `self.embed_dims` should be initialized. For  `XTransformerEncoderLayer._init_layers()` and `XTransformerDecoderLayer._init_layers()`, the attributes of various modules, such as `self.self_attn`, `self.cross_attn`, `self.norms`, `self.ffn`, and `self.embed_dims` should be initialized.

#### Implement detector

The new DETR may be based on an existing DETR. Therefore, we can analyze the differences between them. Inherit the existing detector class or `DetectionTransformer` class. Then override certain functions.

First, the initialization section should be written: The `_init_layers()` is re-written to initialize the modules, including `self.encoder`, `self.decoder`, `self.position_encoding`, and so on. The `init_weights()` is re-written to initialize the weights of the modules.

Then, the forward process should be written: The users are recommended to adopt the well-designed general forward process. Re-write `pre_transformer()`，`forward_encoder()`，`pre_decoder()`，`forward_decoder()`. In `pre_transformer()`, write the logic of generating padding masks and position encoding, and the logic of feature convert from image format into sequence format. At last, return two keyword dictionaries of parameters required by `forward_encoder()` and `forward_decoder()`, respectively. In `pre_decoder()`, write the logic of processing decoder outputs and obtaining decoder queries `query` and their positional embeddings `query_pos`. Finally, return two keyword dictionaries of parameters required by `forward_decoder()` and the function of `self.bbox_head`, respectively.  In `forward_encoder()` and `forward_decoder()`, implement the logic of converting between naming schemes and processing with `self.encoder` or `self.decoder`.

If the users do not adopt the provided process, the `forward_transformer()` should be re-written to implement the forward process of Transformer, and return a keyword dictionary containing the input parameters required by the function of `self.heads`. Or re-write the `loss()`, `prediction()`, `_forward()` completely.

#### Implement head

The new head class should inherit `DETRHead` class. The implementation are divided into the initialization part, the forward part, the loss calculation part and post-processing part:

The initialization of head also requires new `_init_layers()` and `init_weights()`. The `forward()` should be re-written to obtain classification and regression results from decoder outputs. The `loss_by_feat()` transforms the prediction results into loss dictionary for training, and the `prediction_by_feat()` conduct post-processing. If there is added or deleted parameters, the `loss()` and `predict()` may require to be modified.

#### Example: Implement Conditional DETR

Compared with DETR, the main improvement of Conditional DETR are in decoder section.

We can reuse `DetrTransformerEncoder` and `DetrTransformerEncoderLayer`. We require to inherit `DetrTransformerDecoder` and `DetrTransformerDecoderLayers` to write new modules. The two modules both requires new `_init_layers()` and new `forward()`. The implementation can be found at mmdet/models/layers/transformer/conditional_detr_layers.py.

The detector of Conditional DETR is almost same with DETR. There is only small difference, i.e. an extra returned parameter `references` for the decoder. Hence, only `forward_decoder()` is re-written. Moreover, in `_init_layers()`, the decoder module should be replaced with the new module. The implementation can be found at mmdet/models/detector/conditional_detr.py.

For the box head: Since the Conditional DETR uses focal loss, the `init_weights()` is written to modify the initialization of classification head. The `forward()` is re-written to implement the logic of adding predicted offsets to reference. Moreover, `loss()`, `predict()`, `loss_and_prediction()` are re-written to receive the new parameter `references`.
