# X-Decoder

> [X-Decoder: Generalized Decoding for Pixel, Image, and Language](https://arxiv.org/pdf/2212.11270.pdf)

<!-- [ALGORITHM] -->

## Abstract

We present X-Decoder, a generalized decoding model that can predict pixel-level segmentation and language tokens seamlessly. X-Decodert takes as input two types of queries: (i) generic non-semantic queries and (ii) semantic queries induced from text inputs, to decode different pixel-level and token-level outputs in the same semantic space. With such a novel design, X-Decoder is the first work that provides a unified way to support all types of image segmentation and a variety of vision-language (VL) tasks. Further, our design enables seamless interactions across tasks at different granularities and brings mutual benefits by learning a common and rich pixel-level visual-semantic understanding space, without any pseudo-labeling. After pretraining on a mixed set of a limited amount of segmentation data and millions of image-text pairs, X-Decoder exhibits strong transferability to a wide range of downstream tasks in both zero-shot and finetuning settings. Notably, it achieves (1) state-of-the-art results on open-vocabulary segmentation and referring segmentation on eight datasets; (2) better or competitive finetuned performance to other generalist and specialist models on segmentation and VL tasks; and (3) flexibility for efficient finetuning and novel task composition (e.g., referring captioning and image editing).

![img](https://raw.githubusercontent.com/microsoft/X-Decoder/main/images/teaser_new.png)

## Models and results

### Semantic segmentation on ADE20K

**Prepare dataset**

Prepare your dataset according to the [docs](https://mmsegmentation.readthedocs.io/en/latest/user_guides/2_dataset_prepare.html#ade20k).

**Test Command**

```shell
./tools/dist_test.sh  projects/XDecoder/configs/xdecoder-tiny_zeroshot_semseg.py xdecoder_focalt_best_openseg.pt 8
```

| Model                               | mIoU  |                       Config                       |                                            Download                                             |
| :---------------------------------- | :---: | :------------------------------------------------: | :---------------------------------------------------------------------------------------------: |
| `xdecoder_focalt_best_openseg.pt`\* | 25.13 | [config](configs/xdecoder-tiny_zeroshot_semseg.py) | [model](https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focalt_best_openseg.pt) |
