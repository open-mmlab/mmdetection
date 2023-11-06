# Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

[Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we present an open-set object detector, called Grounding DINO, by marrying Transformer-based detector DINO with grounded pre-training, which can detect arbitrary objects with human inputs such as category names or referring expressions. The key solution of open-set object detection is introducing language to a closed-set detector for open-set concept generalization. To effectively fuse language and vision modalities, we conceptually divide a closed-set detector into three phases and propose a tight fusion solution, which includes a feature enhancer, a language-guided query selection, and a cross-modality decoder for cross-modality fusion. While previous works mainly evaluate open-set object detection on novel categories, we propose to also perform evaluations on referring expression comprehension for objects specified with attributes. Grounding DINO performs remarkably well on all three settings, including benchmarks on COCO, LVIS, ODinW, and RefCOCO/+/g. Grounding DINO achieves a 52.5 AP on the COCO detection zero-shot transfer benchmark, i.e., without any training data from COCO. It sets a new record on the ODinW zero-shot benchmark with a mean 26.1 AP.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/42299757/0ed51aeb-3d53-42d8-8563-f6d21364ac95"/>
</div>

## Installation

```shell
cd $MMDETROOT

# source installation
pip install -r requirements/multimodal.txt

# or mim installation
mim install mmdet[multimodal]
```

## NOTE

Grounding DINO utilizes BERT as the language model, which requires access to https://huggingface.co/. If you encounter connection errors due to network access, you can download the required files on a computer with internet access and save them locally. Finally, modify the `lang_model_name` field in the config to the local path. Please refer to the following code:

```python
from transformers import BertConfig, BertModel
from transformers import AutoTokenizer

config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, config=config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

config.save_pretrained("your path/bert-base-uncased")
model.save_pretrained("your path/bert-base-uncased")
tokenizer.save_pretrained("your path/bert-base-uncased")
```

## Inference

```
cd $MMDETROOT

wget https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth

python demo/image_demo.py \
	demo/demo.jpg \
	configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py \
	--weights groundingdino_swint_ogc_mmdet-822d7e9d.pth \
	--texts 'bench . car .'
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/42299757/3a3bd6f1-e2ed-43d4-aa22-0bb07ee6f20b"/>
</div>

## Results and Models

|       Model        | Backbone |   Style   |  COCO mAP  | Official COCO mAP |                  Pre-Train Data                  |                             Config                             |                                                                                                                                                                                                                                         Download                                                                                                                                                                                                                                          |
| :----------------: | :------: | :-------: | :--------: | :---------------: | :----------------------------------------------: | :------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Grounding DINO-T  |  Swin-T  | Zero-shot |    48.5    |       48.4        |                 O365,GoldG,Cap4M                 | [config](grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py) |                                                                                                                                                                                    [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth)                                                                                                                                                                                     |
|  Grounding DINO-T  |  Swin-T  | Finetune  | 58.1(+0.9) |       57.2        |                 O365,GoldG,Cap4M                 |   [config](grounding_dino_swin-t_finetune_16xb2_1x_coco.py)    | [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco/grounding_dino_swin-t_finetune_16xb2_1x_coco_20230921_152544-5f234b20.pth)                                                                                                \| [log](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco/grounding_dino_swin-t_finetune_16xb2_1x_coco_20230921_152544.log.json) |
|  Grounding DINO-B  |  Swin-B  | Zero-shot |    56.9    |       56.7        | COCO,O365,GoldG,Cap4M,OpenImage,ODinW-35,RefCOCO |     [config](grounding_dino_swin-b_pretrain_mixeddata.py)      |                                                                                                                                                                                  [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth)                                                                                                                                                                                   |
|  Grounding DINO-B  |  Swin-B  | Finetune  |    59.7    |                   | COCO,O365,GoldG,Cap4M,OpenImage,ODinW-35,RefCOCO |   [config](grounding_dino_swin-b_finetune_16xb2_1x_coco.py)    |                                               [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco/grounding_dino_swin-b_finetune_16xb2_1x_coco_20230921_153201-f219e0c0.pth)   \| [log](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco/grounding_dino_swin-b_finetune_16xb2_1x_coco_20230921_153201.log.json)                                                |
| Grounding DINO-R50 |   R50    |  Scratch  | 48.9(+0.8) |       48.1        |                                                  |      [config](grounding_dino_r50_scratch_8xb2_1x_coco.py)      |                                                                                          [model](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_coco/grounding_dino_r50_scratch_1x_coco-fe0002f2.pth)  \| [log](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_coco/20230922_114218.json)                                                                                           |

Note:

1. The weights corresponding to the zero-shot model are adopted from the official weights and converted using the [script](../../tools/model_converters/groundingdino_to_mmdet.py). We have not retrained the model for the time being.
2. Finetune refers to fine-tuning on the COCO 2017 dataset. The R50 model is trained using 8 NVIDIA GeForce 3090 GPUs, while the remaining models are trained using 16 NVIDIA GeForce 3090 GPUs. The GPU memory usage is approximately 8.5GB.
3. Our performance is higher than the official model due to two reasons: we modified the initialization strategy and introduced a log scaler.

## Custom Dataset

To facilitate fine-tuning on custom datasets, we use a simple cat dataset as an example, as shown in the following steps.

### 1. Dataset Preparation

```shell
cd mmdetection
wget https://download.openmmlab.com/mmyolo/data/cat_dataset.zip
unzip cat_dataset.zip -d data/cat/
```

cat dataset is a single-category dataset with 144 images, which has been converted to coco format.

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205423220-c4b8f2fd-22ba-4937-8e47-1b3f6a8facd8.png" alt="cat dataset"/>
</div>

### 2. Config Preparation

Due to the simplicity and small number of cat datasets, we use 8 cards to train 20 epochs, scale the learning rate accordingly, and do not train the language model, only the visual model.

The Details of the configuration can be found in [grounding_dino_swin-t_finetune_8xb2_20e_cat](grounding_dino_swin-t_finetune_8xb2_20e_cat.py)

### 3. Visualization and Evaluation

Due to the Grounding DINO is an open detection model, so it can be detected and evaluated even if it is not trained on the cat dataset.

The single image visualization is as follows:

```shell
cd mmdetection
python demo/image_demo.py data/cat/images/IMG_20211205_120756.jpg configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py --weights https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth --texts cat.
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/89173261-16f1-4fd9-ac63-8dc2dcda6616" alt="cat dataset"/>
</div>

The test dataset evaluation on single card is as follows:

```shell
python tools/test.py configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth
```

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.867
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.931
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.867
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.903
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.907
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.907
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.907
```

### 4. Model Training and Visualization

```shell
./tools/dist_train.sh configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py 8 --work-dir cat_work_dir
```

The model will be saved based on the best performance on the test set. The performance of the best model (at epoch 16) is as follows:

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.905
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.923
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.905
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.927
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.937
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.937
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.937
```

We can find that after fine-tuning training, the training of the cat dataset is increased from 86.7 to 90.5.

If we do single image inference visualization again, the result is as follows:

```shell
cd mmdetection
python demo/image_demo.py data/cat/images/IMG_20211205_120756.jpg configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py --weights cat_work_dir/best_coco_bbox_mAP_epoch_16.pth --texts cat.
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/5a027b00-8adb-4283-a47b-2f7a0a2c96d4" alt="cat dataset"/>
</div>
