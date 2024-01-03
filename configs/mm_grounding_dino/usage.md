# Usage

## Install

After installing MMDet according to the instructions in the [get_started](../../docs/zh_cn/get_started.md) section, you need to install additional dependency packages:

```shell
cd $MMDETROOT

pip install -r requirements/multimodal.txt
pip install emoji ddd-dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git"
```

Please note that since the LVIS third-party library does not currently support numpy 1.24, ensure that your numpy version meets the requirements. It is recommended to install numpy version 1.23.

## Instructions

### Download BERT Weight

MM Grounding DINO uses BERT as its language model and requires access to https://huggingface.co/. If you encounter connection errors due to network access issues, you can download the necessary files on a computer with network access and save them locally. Finally, modify the `lang_model_name` field in the configuration file to the local path. For specific instructions, please refer to the following code:

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

### Download NLTK Weight

When MM Grounding DINO performs Phrase Grounding inference, it may extract noun phrases. Although it downloads specific models at runtime, considering that some users' running environments cannot connect to the internet, it is possible to download them in advance to the `~/nltk_data` path.

```python
import nltk
nltk.download('punkt', download_dir='~/nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='~/nltk_data')
```

### Download MM Grounding DINO-T Weight

For convenience in demonstration, you can download the MM Grounding DINO-T model weights in advance to the current path.

```shell
wget load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth' # noqa
```

## Inference

Before inference, for a better experience of the inference effects on different images, it is recommended that you first download [these images](https://github.com/microsoft/X-Decoder/tree/main/inference_demo/images) to the current path.

MM Grounding DINO supports four types of inference methods: Closed-Set Object Detection, Open Vocabulary Object Detection, Phrase Grounding, and Referential Expression Comprehension. The details are explained below.

**(1) Closed-Set Object Detection**

Since MM Grounding DINO is a pretrained model, it can theoretically be applied to any closed-set detection dataset. Currently, we support commonly used datasets such as coco/voc/cityscapes/objects365v1/lvis, etc. Below, we will use coco as an example.

```shell
python demo/image_demo.py images/animals.png \
        configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
        --texts '$: coco'
```

The predictions for `outputs/vis/animals.png` will be generated in the current directory, as shown in the following image.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/1659211c-c117-4097-a659-84ab26efa2d3" width="70%"/>
</div>

Since ostrich is not one of the 80 classes in COCO, it will not be detected.

It's important to note that Objects365v1 and LVIS have a large number of categories. If you try to input all category names directly into the network, it may exceed 256 tokens, leading to poor model predictions. In such cases, you can use the `--chunked-size` parameter to perform chunked predictions. However, please be aware that chunked predictions may take longer to complete due to the large number of categories.

```shell
python demo/image_demo.py images/animals.png \
        configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
        --texts '$: lvis'  --chunked-size 70 \
        --palette random
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/93554cf5-a1c5-4318-8e16-615cd2270fb6" width="70%"/>
</div>

Different `--chunked-size` values can lead to different prediction results. You can experiment with different chunked sizes to find the one that works best for your specific task and dataset.

**(2) Open Vocabulary Object Detection**

Open vocabulary object detection refers to the ability to input arbitrary class names during inference.

```shell
python demo/image_demo.py images/animals.png \
        configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
        --texts 'zebra. giraffe' -c
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/75e4a81f-4644-4306-8f66-60e684ac32db" width="70%"/>
</div>

**(3) Phrase Grounding**

Phrase Grounding refers to the process where a user inputs a natural language description, and the model automatically detects the corresponding bounding boxes for the mentioned noun phrases. It can be used in two ways:

1. Automatically extracting noun phrases using the NLTK library and then performing detection.

```shell
python demo/image_demo.py images/apples.jpg \
        configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
        --texts 'There are many apples here.'
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/7c5839d2-3266-49e1-8be6-012f258d710b" width="70%"/>
</div>

The program will automatically split `many apples` as a noun phrase and then detect the corresponding objects. Different input descriptions can have a significant impact on the prediction results.

2. Users can manually specify which parts of the sentence are noun phrases to avoid errors in NLTK extraction.

```shell
python demo/image_demo.py images/fruit.jpg \
        configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
        --texts 'The picture contains watermelon, flower, and a white bottle.' \
        --tokens-positive "[[[21,31]], [[45,59]]]"  --pred-score-thr 0.12
```

The noun phrase corresponding to positions 21-31 is `watermelon`, and the noun phrase corresponding to positions 45-59 is `a white bottle`.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/15080faf-048d-4201-a126-a9c773580f5e" width="70%"/>
</div>

**(4) Referential Expression Comprehension**

Referential expression understanding refers to the model automatically comprehending the referential expressions involved in a user's language description without the need for noun phrase extraction.

```shell
python demo/image_demo.py images/apples.jpg \
        configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
        --texts 'red apple.' \
        --tokens-positive -1
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/40b970c3-60cd-4c78-a2cb-2c41b0442932" width="70%"/>
</div>

## Evaluation

Our provided evaluation scripts are unified, and you only need to prepare the data in advance and then run the relevant configuration.

(1) Zero-Shot COCO2017 val

```shell
# single GPU
python tools/test.py configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth

# 8 GPUs
./tools/dist_test.sh configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth 8
```

(2) Zero-Shot ODinW13

```shell
# single GPU
python tools/test.py configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw13.py \
        grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth

# 8 GPUs
./tools/dist_test.sh configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw13.py \
        grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth 8
```

## Visualization of Evaluation Results

For the convenience of visualizing and analyzing model prediction results, we provide support for visualizing evaluation dataset prediction results. Taking referential expression understanding as an example, the usage is as follows:

```shell
python tools/test.py configs/mm_grounding_dino/refcoco/grounding_dino_swin-t_pretrain_zeroshot_refexp \
        grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth --work-dir refcoco_result --show-dir save_path
```

During the inference process, it will save the visualization results to the `refcoco_result/{current_timestamp}/save_path` directory. For other evaluation dataset visualizations, you only need to replace the configuration file.

Here are some visualization results for various datasets. The left image represents the Ground Truth (GT). The right image represents the Predicted Result.

1. COCO2017 val Results：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/3a0fa894-c0a5-4c1f-bdf0-1c6fd17abafa" width="70%"/>
</div>

2. Flickr30k Entities Results：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/e9f2667f-9dca-464b-b995-599aa2731b34" width="70%"/>
</div>

3. DOD Results：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/c71a306b-1055-4344-ba1d-ae4c57f2cb2f" width="70%"/>
</div>

4. RefCOCO val Results：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/b175959d-d788-4b5e-8b11-e8e34753457f" width="70%"/>
</div>

5. RefCOCO testA Results：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/c087f889-f96c-4355-8a15-7dc2738b4223" width="70%"/>
</div>

6. gRefCOCO val Results：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/96c2e783-17da-462e-a7cf-937555e26c90" width="70%"/>
</div>

## Training

If you want to reproduce our results, you can train the model by using the following command after preparing the dataset:

```shell
# Training on a single machine with 8 GPUs for obj365v1 dataset
./tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py 8
# Training on a single machine with 8 GPUs for datasets like obj365v1, goldg, grit, v3det, and other datasets is similar.
./tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py 8
```

For multi-machine training, please refer to [train.md](../../docs/zh_cn/user_guides/train.md). The MM-Grounding-DINO T model is designed to work with 32 GPUs (specifically, 3090Ti GPUs). If your total batch size is not 32x4=128, you will need to manually adjust the learning rate accordingly.

### Pretraining Custom Format Explanation

In order to standardize the pretraining formats for different datasets, we refer to the format design proposed by [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino). Specifically, it is divided into two formats.

**(1) Object Detection Format (OD)**

```text
{"filename": "obj365_train_000000734304.jpg",
 "height": 512,
 "width": 769,
 "detection": {
    "instances": [
          {"bbox": [109.4768676992, 346.0190429696, 135.1918335098, 365.3641967616], "label": 2, "category": "chair"},
          {"bbox": [58.612365705900004, 323.2281494016, 242.6005859067, 451.4166870016], "label": 8, "category": "car"}
                ]
      }
}
```

The numerical values corresponding to labels in the label dictionary should match the respective label_map. Each item in the instances list corresponds to a bounding box (in the format x1y1x2y2).

**(2) Phrase Grounding Format (VG)**

```text
{"filename": "2405116.jpg",
 "height": 375,
 "width": 500,
 "grounding":
     {"caption": "Two surfers walking down the shore. sand on the beach.",
      "regions": [
            {"bbox": [206, 156, 282, 248], "phrase": "Two surfers", "tokens_positive": [[0, 3], [4, 11]]},
            {"bbox": [303, 338, 443, 343], "phrase": "sand", "tokens_positive": [[36, 40]]},
            {"bbox": [[327, 223, 421, 282], [300, 200, 400, 210]], "phrase": "beach", "tokens_positive": [[48, 53]]}
               ]
      }
```

The `tokens_positive` field indicates the character positions of the current phrase within the caption.

## Example of Fine-tuning Custom Dataset

In order to facilitate downstream fine-tuning on custom datasets, we have provided a fine-tuning example using the simple "cat" dataset as an illustration.

### 1 Data Preparation

```shell
cd mmdetection
wget https://download.openmmlab.com/mmyolo/data/cat_dataset.zip
unzip cat_dataset.zip -d data/cat/
```

The "cat" dataset is a single-category dataset consisting of 144 images, already converted to the COCO format.

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205423220-c4b8f2fd-22ba-4937-8e47-1b3f6a8facd8.png" alt="cat dataset"/>
</div>

### 2 Configuration Preparation

Due to the simplicity and small size of the "cat" dataset, we trained it for 20 epochs using 8 GPUs, with corresponding learning rate scaling. We did not train the language model, only the visual model.

Detailed configuration information can be found in [grounding_dino_swin-t_finetune_8xb4_20e_cat](grounding_dino_swin-t_finetune_8xb4_20e_cat.py).

### 3 Visualization and Evaluation of Zero-Shot Results

Due to MM Grounding DINO being an open-set detection model, you can perform detection and evaluation even if it was not trained on the cat dataset.

Visualization of a single image:

```shell
cd mmdetection
python demo/image_demo.py data/cat/images/IMG_20211205_120756.jpg configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth --texts cat.
```

Evaluation results of Zero-shot on test dataset：

```shell
python tools/test.py configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
```

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.881
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.929
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.881
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.913
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.913
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.913
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.913
```

### 4 Fine-tuning

```shell
./tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py 8 --work-dir cat_work_dir
```

The model will save the best-performing checkpoint. It achieved its best performance at the 16th epoch, with the following results:

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.901
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.930
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.901
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.967
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.967
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.967
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.967
```

We can observe that after fine-tuning, the training performance on the cat dataset improved from 88.1 to 90.1. However, due to the small dataset size, the evaluation metrics show some fluctuations.

## Iterative Generation and Optimization Pipeline of Model Self-training Pseduo Label

To facilitate users in creating their own datasets from scratch or those who want to leverage the model's inference capabilities for iterative pseudo-label generation and optimization, continuously modifying pseudo-labels to improve model performance, we have provided relevant pipelines.

Since we have defined two data formats, we will provide separate explanations for demonstration purposes.

### 1 Object Detection Format

Here, we continue to use the aforementioned cat dataset as an example. Let's assume that we currently have a series of images and predefined categories but no annotations.

1. Generate initial `odvg` format file

```python
import os
import cv2
import json
import jsonlines

data_root = 'data/cat'
images_path = os.path.join(data_root, 'images')
out_path = os.path.join(data_root, 'cat_train_od.json')
metas = []
for files in os.listdir(images_path):
    img = cv2.imread(os.path.join(images_path, files))
    height, width, _ = img.shape
    metas.append({"filename": files, "height": height, "width": width})

with jsonlines.open(out_path, mode='w') as writer:
    writer.write_all(metas)

# 生成 label_map.json，由于只有一个类别，所以只需要写一个 cat 即可
label_map_path = os.path.join(data_root, 'cat_label_map.json')
with open(label_map_path, 'w') as f:
    json.dump({'0': 'cat'}, f)
```

Two files, `cat_train_od.json` and `cat_label_map.json`, will be generated in the `data/cat` directory.

2. Inference with pre-trained model and save the results

We provide a readily usable [configuration](grounding_dino_swin-t_pretrain_pseudo-labeling_cat.py). If you are using a different dataset, you can refer to this configuration for modifications.

```shell
python tools/test.py configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_pseudo-labeling_cat.py \
    grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
```

A new file `cat_train_od_v1.json` will be generated in the `data/cat` directory. You can manually open it to confirm or use the provided [script](../../tools/analysis_tools/browse_grounding_raw.py) to visualize the results.

```shell
python tools/analysis_tools/browse_grounding_raw.py data/cat/ cat_train_od_v1.json images --label-map-file cat_label_map.json -o your_output_dir --not-show
```

The visualization results will be generated in the `your_output_dir` directory.

3. Continue training to boost performance

After obtaining pseudo-labels, you can mix them with some pre-training data for further pre-training to improve the model's performance on the current dataset. Then, you can repeat step 2 to obtain more accurate pseudo-labels, and continue this iterative process.

### 2 Phrase Grounding Format

1. Generate initial `odvg` format file

The bootstrapping process of Phrase Grounding requires providing captions corresponding to each image and pre-segmented phrase information initially. Taking flickr30k entities images as an example, the generated typical file should look like this:

```text
[
{"filename": "3028766968.jpg",
 "height": 375,
 "width": 500,
 "grounding":
     {"caption": "Man with a black shirt on sit behind a desk sorting threw a giant stack of people work with a smirk on his face .",
      "regions": [
                 {"bbox": [0, 0, 1, 1], "phrase": "a giant stack of people", "tokens_positive": [[58, 81]]},
                 {"bbox": [0, 0, 1, 1], "phrase": "a black shirt", "tokens_positive": [[9, 22]]},
                 {"bbox": [0, 0, 1, 1], "phrase": "a desk", "tokens_positive": [[37, 43]]},
                 {"bbox": [0, 0, 1, 1], "phrase": "his face", "tokens_positive": [[103, 111]]},
                 {"bbox": [0, 0, 1, 1], "phrase": "Man", "tokens_positive": [[0, 3]]}]}}
{"filename": "6944134083.jpg",
 "height": 319,
 "width": 500,
 "grounding":
    {"caption": "Two men are competing in a horse race .",
    "regions": [
                {"bbox": [0, 0, 1, 1], "phrase": "Two men", "tokens_positive": [[0, 7]]}]}}
]
```

Bbox needs to be set to `[0, 0, 1, 1]` for initialization to make sure the programme could run, but this value would not be utilized.

```text
{"filename": "3028766968.jpg", "height": 375, "width": 500, "grounding": {"caption": "Man with a black shirt on sit behind a desk sorting threw a giant stack of people work with a smirk on his face .", "regions": [{"bbox": [0, 0, 1, 1], "phrase": "a giant stack of people", "tokens_positive": [[58, 81]]}, {"bbox": [0, 0, 1, 1], "phrase": "a black shirt", "tokens_positive": [[9, 22]]}, {"bbox": [0, 0, 1, 1], "phrase": "a desk", "tokens_positive": [[37, 43]]}, {"bbox": [0, 0, 1, 1], "phrase": "his face", "tokens_positive": [[103, 111]]}, {"bbox": [0, 0, 1, 1], "phrase": "Man", "tokens_positive": [[0, 3]]}]}}
{"filename": "6944134083.jpg", "height": 319, "width": 500, "grounding": {"caption": "Two men are competing in a horse race .", "regions": [{"bbox": [0, 0, 1, 1], "phrase": "Two men", "tokens_positive": [[0, 7]]}]}}
```

You can directly copy the text above, and assume that the text content is pasted into a file named `flickr_simple_train_vg.json`, which is placed in the pre-prepared `data/flickr30k_entities` dataset directory, as detailed in the data preparation document.

2. Inference with pre-trained model and save the results

We provide a directly usable [configuration](https://chat.openai.com/c/grounding_dino_swin-t_pretrain_pseudo-labeling_flickr30k.py). If you are using a different dataset, you can refer to this configuration for modifications.

```shell
python tools/test.py configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_pseudo-labeling_flickr30k.py \
    grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
```

The translation of your text from Chinese to English is: "A new file `flickr_simple_train_vg_v1.json` will be generated in the `data/flickr30k_entities` directory. You can manually open it to confirm or use the [script](../../tools/analysis_tools/browse_grounding_raw.py) to visualize the effects

```shell
python tools/analysis_tools/browse_grounding_raw.py data/flickr30k_entities/ flickr_simple_train_vg_v1.json flickr30k_images -o your_output_dir --not-show
```

The visualization results will be generated in the `your_output_dir` directory, as shown in the following image:

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/a1c72d52-fa52-4ebe-b793-716d34e7b83f" width="50%"/>
</div>

3. Continue training to boost performance

After obtaining the pseudo-labels, you can mix some pre-training data to continue pre-training jointly, which enhances the model's performance on the current dataset. Then, rerun step 2 to obtain more accurate pseudo-labels, and repeat this cycle iteratively.
