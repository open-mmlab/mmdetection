# Detection Transformer

Detection Transformers (DETRs) are a series of Transformer-based object detectors.

In most DETRs, the features extracted from backbone and neck were fed into a Transformer which is composed of an encoder and a decoder. The Transformer directly outputs a set of queries in parallel. Each query corresponds to one prediction, which may be an object or `no object`.

![DETR_overall](.\DETR_overall.png)

There is a brand new implementation of DETRs in MMDetection, whose features are fourfold:

#### 1. A unified base detector class for DETRs.

According to the existing DETRs, a unified base detector class `DetectionTransformer` is designed to standardize the architectures and forward procedures of DETRs. (mmdet/models/detectors/base_detr.py)

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

***( Add a figure here, to illustrate single-scale feature \<--> sequence feature***

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

***( Add a figure here, to illustrate multi-scale features \<--> sequence feature, what is spatial_shape and lvl_start_index)***

#### Positional embedding of DETRs

For the sequence data, both the position of each element in the sequence and the arrangement of the elements are essential for the semantic characterization of the data; For the image data, both the position of each pixel on the whole image and the spatial arrangement of the pixels are also critical for understanding image semantic information.

The attention calculation procedure of Transformer is usually as follow: a query embedding calculates self attentions with the key embeddings. Then, the value embeddings are weighted with the attentions and aggregated into the output embedding. In the attention calculations, each element in the sequence is independent, and their position information is lost. Besides, the attention calculations are permutation-invariant. Therefore, the position encodings are usually embedded for queries, keys, and values before calculating attentions as positional embeddings.

There are also positional embeddings for the inputs of attention modules in DETRs. Unlike most cases, DETRs only embed for queries and keys, and not embed for values. Moreover, DETRs embed positions of both spatial directions, i.e. row and column, namely 2D position encoding.

![](C:\Users\lqy\Desktop\doc_detr\DETR_positional_encoding.png)

The left sub-figure illustrates the 2D position encoding process: The positional embeddings for queries or keys have same embedding dimension with the queries or keys. In the 2D position encoding of DETRs, the dimension of `C` is divided into two partitions of `C/2` uniformly. The former one is embedded for row position and the latter one is embedded for the column position.

The right sub-figure is excerpted from DAB-DETR paper ( may require to re-paint ), which illustrates the positional embeddings of DETRs. The queries `Q` and keys `K` are all composed of two partitions: content queries / keys which attend to object feature content, and positional queries / keys which attend to positional information. The values `V` have not positional partitions. For encoder: content queries, content keys, and values are all from image features.  Positional queries and positional keys are from the 2D position embeddings of image features. For decoder: content keys, values, and positional keys remain. Content queries are from the outputs of last decoder layer or the initial decoder queries. Positional queries are positional embeddings of the decoder queries.

#### Object detection paradigm of set prediction

Most DETRs were set prediction-based detectors. They eliminated the complicated components required by many convenient detectors, such as non-maximum suppression, anchor generation. They obtains a set of predictions, and each prediction includes a category and a bounding box. The classification results include all object categories and a  `no object`  class.

In training: the Hungarian algorithm is used to assign a prediction to each ground truth target. Both classification losses and boxes losses are calculated for the predictions which have been assigned with certain ground truth. While for the predictions which have not been assigned with any ground truth. Their target labels are then assigned with  `no object`  class and only classification losses are calculated.

In inference: the final detection results can be obtained by directly remove the predictions whose classification results are  `no object`  and predictions whose classification confidences are lower than a presupposed threshold, without complicated post-processing.

### Appointment

#### Parameter names

In various codebase of DETRs, there are multiple meanings for the parameter name `query`, which ......

#### Unified data flow

### Customize a DETR

#### Implement Transformer components

#### Implement detector

#### Implement head

#### Example: Implement DINO
