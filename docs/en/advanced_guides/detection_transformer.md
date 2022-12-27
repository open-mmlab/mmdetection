# Detection Transformer

Detection Transformers (DETRs) are a series of Transformer-based object detectors.

In most DETRs, the features extracted from backbone and neck were fed into a Transformer which is composed of an encoder and a decoder. The Transformer directly outputs a set of queries in parallel. Each query corresponds to one prediction, which may be an object or `no object`.

![DETR_overall](.\DETR_overall.png)

There is a brand new implementation of DETRs in MMDetection, whose features are fourfold:

#### 1. A unified base detector class for DETRs.

According to the existing DETRs, a unified base detector class `DetectionTransformer` is designed to standardize the architectures and forward procedures of DETRs. (mmdet/models/detectors/base_detr.py)

Each detector inherited from `DetectionTransformer` is mainly consists of six components: `backbone`, `neck`, `positional_encoding`, `encoder`, `decoder`, and `bbox_head`. The `backbone` and `neck` extract features of input batch. The `positional_encoding` encodes the absolute position of each feature points. The `encoder` processes output features of neck. The `decoder` pools encoder features of objects into the queries. At last, the `bbox_head` makes predictions on output queries of decoder.

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

The original intricate modules of Transformer, including encoder, decoder, encoder layer, and decoder layer, are replaced with straightforward modules which attend to clear logic rather than excessive compatibility. Additionally, the Transformer components are moved out of the head. Hence, the head becomes lightweight and focuses on getting detection results and calculating losses.

#### 3. Code with better readability and usability

The code of all DETR-related modules are refactored to enhance the readability and usability. Specifically, Firstly, the refactored modules have reasonable designing and uniform implementation, which makes it easy to read the code of supported DETRs and implement a custom DETR. Secondly, The overused registration mechanism for building modules are replaced with direct initialization, which benefits code reading and jumping. At last, the new modules come with more detailed docstring and comments that attempt to help the user understand the them.

#### 4. More SOTA DETRs

In addition to the supported DETR and Deformable DETR, three SOTA DETRs are added, including Conditional DETR, DAB DETR, and DINO. The pre-trained weights has been available in the model zoo.

### Appointment
