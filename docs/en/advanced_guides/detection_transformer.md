# Detection Transformer

Detection Transformers (DETRs) are a series of Transformer-based object detectors. In a DETR, an encoder is used to process output features of neck, then several queries interact with the output features of encoder using a decoder and do the regression and classification with the bounding box head. 
The code of DETRs in MMDetection are refactored. The features of the DETRs in the new MMDetection are fourfold:

#### 1. A unified base detector class for DETRs.
According to the existing DETRs, a unified base detector class `DetectionTransformer` is designed to standardize the architectures and forward procedures of DETRs. Each detector inherited from `DetectionTransformer` is mainly consists of five components, i.e. backbone, neck, encoder, decoder, head. Besides, the whole forward process includes six steps: extracting features, preparing for transformer, forward with encoder, preparing for decoder, forward with decoder, and calculate with head.

#### 2. More concise components.
The modules of Transformer, and head become more concise. The original intricate modules of Transformer, including encoder, decoder, encoder layer, and decoder layer, are replaced with straightforward modules which attend to clear logic rather than excessive compatibility. Additionally, the Transformer components are moved out of the head. Hence, the head becomes lightweight and focuses on getting detection results and calculating losses.

#### 3. Code with better readability and usability

The code of all DETR-related modules are refactored to enhance the readability and usability. Specifically, Firstly, the refactored modules have reasonable designing and uniform implementation, which makes it easy to read the code of supported DETRs and implement a custom DETR. Secondly, The overused registration mechanism for building modules are replaced with direct initialization, which benefits code reading and jumping. At last, the new modules come with more detailed docstring and comments that attempt to help the user understand the them.

#### 4. More SOTA DETRs

In addition to the supported DETR and Deformable DETR, three SOTA DETRs are added, including Conditional DETR, DAB DETR, and DINO. The pre-trained weights has been available in the model zoo.

