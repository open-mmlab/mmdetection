# X-Decoder

> [X-Decoder: Generalized Decoding for Pixel, Image, and Language](https://arxiv.org/pdf/2212.11270.pdf)

<!-- [ALGORITHM] -->

## Abstract

We present X-Decoder, a generalized decoding model that can predict pixel-level segmentation and language tokens seamlessly. X-Decodert takes as input two types of queries: (i) generic non-semantic queries and (ii) semantic queries induced from text inputs, to decode different pixel-level and token-level outputs in the same semantic space. With such a novel design, X-Decoder is the first work that provides a unified way to support all types of image segmentation and a variety of vision-language (VL) tasks. Further, our design enables seamless interactions across tasks at different granularities and brings mutual benefits by learning a common and rich pixel-level visual-semantic understanding space, without any pseudo-labeling. After pretraining on a mixed set of a limited amount of segmentation data and millions of image-text pairs, X-Decoder exhibits strong transferability to a wide range of downstream tasks in both zero-shot and finetuning settings. Notably, it achieves (1) state-of-the-art results on open-vocabulary segmentation and referring segmentation on eight datasets; (2) better or competitive finetuned performance to other generalist and specialist models on segmentation and VL tasks; and (3) flexibility for efficient finetuning and novel task composition (e.g., referring captioning and image editing).

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/cb126615-9402-4c19-8ea9-133722d7519c" width="70%"/>
</div>

## Installation

```shell
# if source
pip install -r requirements/multimodal.txt

# if wheel
mim install mmdet[multimodal]
```

## How to use it?

For convenience, you can download the weights to the `mmdetection` root dir

```shell
wget https://download.openmmlab.com/mmdetection/v3.0/xdecoder/xdecoder_focalt_last_novg.pt
wget https://download.openmmlab.com/mmdetection/v3.0/xdecoder/xdecoder_focalt_best_openseg.pt
```

The above two weights are directly copied from the official website without any modification. The specific source is https://github.com/microsoft/X-Decoder

For convenience of demonstration, please download [the folder](https://github.com/microsoft/X-Decoder/tree/main/inference_demo/images) and place it in the root directory of mmdetection.

**(1) Open Vocabulary Semantic Segmentation**

```shell
cd projects/XDecoder
python demo.py ../../images/animals.png configs/xdecoder-tiny_zeroshot_open-vocab-semseg_coco.py --weights ../../xdecoder_focalt_last_novg.pt --texts zebra.giraffe
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/c397c0ed-859a-4004-8725-78a591742bc8" width="70%"/>
</div>

**(2) Open Vocabulary Instance Segmentation**

```shell
cd projects/XDecoder
python demo.py ../../images/owls.jpeg configs/xdecoder-tiny_zeroshot_open-vocab-instance_coco.py --weights ../../xdecoder_focalt_last_novg.pt --texts owl
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/494b0b1c-4a42-4019-97ae-d33ee68af3d2" width="70%"/>
</div>

**(3) Open Vocabulary Panoptic Segmentation**

```shell
cd projects/XDecoder
python demo.py ../../images/street.jpg configs/xdecoder-tiny_zeroshot_open-vocab-panoptic_coco.py --weights ../../xdecoder_focalt_last_novg.pt  --text car.person --stuff-text tree.sky
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/9ad1e0f4-75ce-4e37-a5cc-83e0e8a722ed" width="70%"/>
</div>

**(4) Referring Expression Segmentation**

```shell
cd projects/XDecoder
python demo.py ../../images/fruit.jpg configs/xdecoder-tiny_zeroshot_open-vocab-ref-seg_refcocog.py --weights ../../xdecoder_focalt_last_novg.pt  --text "The larger watermelon. The front white flower. White tea pot."
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/f3ecdb50-20f0-4dc4-aa9c-90995ae04893" width="70%"/>
</div>

**(5) Image Caption**

```shell
cd projects/XDecoder
python demo.py ../../images/penguin.jpeg configs/xdecoder-tiny_zeroshot_caption_coco2014.py --weights ../../xdecoder_focalt_last_novg.pt
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/7690ab79-791e-4011-ab0c-01f46c4a3d80" width="70%"/>
</div>

**(6) Referring Expression Image Caption**

```shell
cd projects/XDecoder
python demo.py ../../images/fruit.jpg configs/xdecoder-tiny_zeroshot_ref-caption.py --weights ../../xdecoder_focalt_last_novg.pt --text 'White tea pot'
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/bae2fdba-0172-4fc8-8ad1-73b54c64ec30" width="70%"/>
</div>

**(7) Text Image Region Retrieval**

```shell
cd projects/XDecoder
python demo.py ../../images/coco configs/xdecoder-tiny_zeroshot_text-image-retrieval.py --weights ../../xdecoder_focalt_last_novg.pt --text 'pizza on the plate'
```

```text
The image that best matches the given text is ../../images/coco/000.jpg and probability is 0.998
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/479de6b2-88e7-41f0-8228-4b9a48f52954" width="70%"/>
</div>

We have also prepared a gradio program in the `projects/gradio_demo` directory, which you can run interactively all the inference supported by mmdetection in your browser.

## Models and results

### Semantic segmentation on ADE20K

Prepare your dataset according to the [docs](../../docs/en/user_guides/dataset_prepare.md#ade20k-2016-dataset-preparation).

**Test Command**

Since semantic segmentation is a pixel-level task, we don't need to use a threshold to filter out low-confidence predictions. So we set `model.test_cfg.use_thr_for_mc=False` in the test command.

```shell
./tools/dist_test.sh projects/XDecoder/configs/xdecoder-tiny_zeroshot_open-vocab-semseg_ade20k.py xdecoder_focalt_best_openseg.pt 8 --cfg-options model.test_cfg.use_thr_for_mc=False
```

| Model                             | mIoU  | mIOU(official) |                                Config                                |
| :-------------------------------- | :---: | :------------: | :------------------------------------------------------------------: |
| `xdecoder_focalt_best_openseg.pt` | 25.24 |     25.13      | [config](configs/xdecoder-tiny_zeroshot_open-vocab-semseg_ade20k.py) |

### Instance segmentation on ADE20K

Prepare your dataset according to the [docs](../../docs/en/user_guides/dataset_prepare.md#ade20k-2016-dataset-preparation).

```shell
./tools/dist_test.sh projects/XDecoder/configs/xdecoder-tiny_zeroshot_open-vocab-instance_ade20k.py xdecoder_focalt_best_openseg.pt 8
```

| Model                             | mIoU | mIOU(official) |                                 Config                                 |
| :-------------------------------- | :--: | :------------: | :--------------------------------------------------------------------: |
| `xdecoder_focalt_best_openseg.pt` | 10.1 |      10.1      | [config](configs/xdecoder-tiny_zeroshot_open-vocab-instance_ade20k.py) |

### Panoptic segmentation on ADE20K

Prepare your dataset according to the [docs](../../docs/en/user_guides/dataset_prepare.md#ade20k-2016-dataset-preparation).

```shell
./tools/dist_test.sh projects/XDecoder/configs/xdecoder-tiny_zeroshot_open-vocab-panoptic_ade20k.py xdecoder_focalt_best_openseg.pt 8
```

| Model                             | mIoU  | mIOU(official) |                                 Config                                 |
| :-------------------------------- | :---: | :------------: | :--------------------------------------------------------------------: |
| `xdecoder_focalt_best_openseg.pt` | 19.11 |     18.97      | [config](configs/xdecoder-tiny_zeroshot_open-vocab-panoptic_ade20k.py) |

### Semantic segmentation on COCO2017

Prepare your dataset according to the [docs](../../docs/en/user_guides/dataset_prepare.md#coco-semantic-dataset-preparation) of `(2) use panoptic dataset` part.

```shell
./tools/dist_test.sh projects/XDecoder/configs/xdecoder-tiny_zeroshot_open-vocab-semseg_coco.py xdecoder_focalt_last_novg.pt 8 --cfg-options model.test_cfg.use_thr_for_mc=False
```

| Model                                           | mIOU | mIOU(official) |                               Config                               |
| :---------------------------------------------- | :--: | :------------: | :----------------------------------------------------------------: |
| `xdecoder-tiny_zeroshot_open-vocab-semseg_coco` | 62.1 |      62.1      | [config](configs/xdecoder-tiny_zeroshot_open-vocab-semseg_coco.py) |

### Instance segmentation on COCO2017

Prepare your dataset according to the [docs](../../docs/en/user_guides/dataset_prepare.md#basic-detection-dataset-preparation).

```shell
./tools/dist_test.sh projects/XDecoder/configs/xdecoder-tiny_zeroshot_open-vocab-instance_coco.py xdecoder_focalt_last_novg.pt 8
```

| Model                                             | Mask mAP | Mask mAP(official) |                                Config                                |
| :------------------------------------------------ | :------: | :----------------: | :------------------------------------------------------------------: |
| `xdecoder-tiny_zeroshot_open-vocab-instance_coco` |   39.8   |        39.7        | [config](configs/xdecoder-tiny_zeroshot_open-vocab-instance_coco.py) |

### Panoptic segmentation on COCO2017

Prepare your dataset according to the [docs](../../docs/en/user_guides/dataset_prepare.md#basic-detection-dataset-preparation).

```shell
./tools/dist_test.sh projects/XDecoder/configs/xdecoder-tiny_zeroshot_open-vocab-panoptic_coco.py xdecoder_focalt_last_novg.pt 8
```

| Model                                             |  PQ   | PQ(official) |                                Config                                |
| :------------------------------------------------ | :---: | :----------: | :------------------------------------------------------------------: |
| `xdecoder-tiny_zeroshot_open-vocab-panoptic_coco` | 51.42 |    51.16     | [config](configs/xdecoder-tiny_zeroshot_open-vocab-panoptic_coco.py) |

### Referring segmentation on RefCOCO

Prepare your dataset according to the [docs](../../docs/en/user_guides/dataset_prepare.md#refcoco-dataset-preparation).

```shell
./tools/dist_test.sh  projects/XDecoder/configs/xdecoder-tiny_zeroshot_open-vocab-ref-seg_refcocog.py xdecoder_focalt_last_novg.pt 8  --cfg-options test_dataloader.dataset.split='val'
```

| Model                          |  text mode   |  cIoU   | cIOU(official) |                                 Config                                  |
| :----------------------------- | :----------: | :-----: | :------------: | :---------------------------------------------------------------------: |
| `xdecoder_focalt_last_novg.pt` | select first | 58.8415 |     57.85      | [config](configs/xdecoder-tiny_zeroshot_open-vocab-ref-seg_refcocog.py) |
| `xdecoder_focalt_last_novg.pt` |   original   | 60.0321 |       -        | [config](configs/xdecoder-tiny_zeroshot_open-vocab-ref-seg_refcocog.py) |
| `xdecoder_focalt_last_novg.pt` |    concat    | 60.3551 |       -        | [config](configs/xdecoder-tiny_zeroshot_open-vocab-ref-seg_refcocog.py) |

**Note:**

1. If you set the scale of `Resize` to (1024, 512), the result will be `57.69`.
2. `text mode` is the `RefCoCoDataset` parameter in MMDetection, it determines the texts loaded to the data list. It can be set to `select_first`, `original`, `concat` and `random`.
   - `select_first`: select the first text in the text list as the description to an instance.
   - `original`: use all texts in the text list as the description to an instance.
   - `concat`: concatenate all texts in the text list as the description to an instance.
   - `random`: randomly select one text in the text list as the description to an instance, usually used for training.

### Image Caption on COCO2014

Prepare your dataset according to the [docs](../../docs/en/user_guides/dataset_prepare.md#coco-caption-dataset-preparation).

Before testing, you need to install jdk 1.8, otherwise it will prompt that java does not exist during the evaluation process

```
./tools/dist_test.sh projects/XDecoder/configs/xdecoder-tiny_zeroshot_caption_coco2014.py xdecoder_focalt_last_novg.pt 8
```

| Model                                     | BLEU-4 | CIDER  |                            Config                            |
| :---------------------------------------- | :----: | :----: | :----------------------------------------------------------: |
| `xdecoder-tiny_zeroshot_caption_coco2014` | 35.26  | 116.81 | [config](configs/xdecoder-tiny_zeroshot_caption_coco2014.py) |

## Citation

```latex
@article{zou2022xdecoder,
  author      = {Zou*, Xueyan and Dou*, Zi-Yi and Yang*, Jianwei and Gan, Zhe and Li, Linjie and Li, Chunyuan and Dai, Xiyang and Wang, Jianfeng and Yuan, Lu and Peng, Nanyun and Wang, Lijuan and Lee*, Yong Jae and Gao*, Jianfeng},
  title       = {Generalized Decoding for Pixel, Image and Language},
  publisher   = {arXiv},
  year        = {2022},
}
```
