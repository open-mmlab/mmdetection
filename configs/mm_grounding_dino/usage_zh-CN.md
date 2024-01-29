# 用法说明

## 安装

在按照 [get_started](../../docs/zh_cn/get_started.md) 一节的说明安装好 MMDet 之后，需要安装额外的依赖包：

```shell
cd $MMDETROOT

pip install -r requirements/multimodal.txt
pip install emoji ddd-dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git"
```

请注意由于 LVIS 第三方库暂时不支持 numpy 1.24，因此请确保您的 numpy 版本符合要求。建议安装 numpy 1.23 版本。

## 说明

### BERT 权重下载

MM Grounding DINO 采用了 BERT 作为语言模型，需要访问 https://huggingface.co/, 如果您因为网络访问问题遇到连接错误，可以在有网络访问权限的电脑上下载所需文件并保存在本地。最后，修改配置文件中的 `lang_model_name` 字段为本地路径即可。具体请参考以下代码：

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

### NLTK 权重下载

MM Grounding DINO 在进行 Phrase Grounding 推理时候可能会进行名词短语提取，虽然会在运行时候下载特定的模型，但是考虑到有些用户运行环境无法联网，因此可以提前下载到 `~/nltk_data` 路径下

```python
import nltk
nltk.download('punkt', download_dir='~/nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='~/nltk_data')
```

### MM Grounding DINO-T 模型权重下载

为了方便演示，您可以提前下载 MM Grounding DINO-T 模型权重到当前路径下

```shell
wget load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth' # noqa
```

## 推理

在推理前，为了更好的体验不同图片的推理效果，建议您先下载 [这些图片](https://github.com/microsoft/X-Decoder/tree/main/inference_demo/images) 到当前路径下

MM Grounding DINO 支持了闭集目标检测，开放词汇目标检测，Phrase Grounding 和指代性表达式理解 4 种推理方式，下面详细说明。

**(1) 闭集目标检测**

由于 MM Grounding DINO 是预训练模型，理论上可以应用于任何闭集检测数据集，目前我们支持了常用的 coco/voc/cityscapes/objects365v1/lvis 等，下面以 coco 为例

```shell
python demo/image_demo.py images/animals.png \
        configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
        --texts '$: coco'
```

会在当前路径下生成 `outputs/vis/animals.png` 的预测结果，如下图所示

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/1659211c-c117-4097-a659-84ab26efa2d3" width="70%"/>
</div>

由于鸵鸟并不在 COCO 80 类中, 因此不会检测出来。

需要注意，由于 objects365v1 和 lvis 类别很多，如果直接将类别名全部输入到网络中，会超过 256 个 token 导致模型预测效果极差，此时我们需要通过 `--chunked-size` 参数进行截断预测, 同时预测时间会比较长。

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

不同的 `--chunked-size` 会导致不同的预测效果，您可以自行尝试。

**(2) 开放词汇目标检测**

开放词汇目标检测是指在推理时候，可以输入任意的类别名

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

Phrase Grounding 是指的用户输入一句语言描述，模型自动对其涉及到的名词短语想对应的 bbox 进行检测，有两种用法

1. 通过 NLTK 库自动提取名词短语，然后进行检测

```shell
python demo/image_demo.py images/apples.jpg \
        configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
        --texts 'There are many apples here.'
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/7c5839d2-3266-49e1-8be6-012f258d710b" width="70%"/>
</div>

程序内部会自动切分出 `many apples` 作为名词短语，然后检测出对应物体。不同的输入描述对预测结果影响很大。

2. 用户自己指定句子中哪些为名词短语，避免 NLTK 提取错误的情况

```shell
python demo/image_demo.py images/fruit.jpg \
        configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
        --texts 'The picture contains watermelon, flower, and a white bottle.' \
        --tokens-positive "[[[21,31]], [[45,59]]]"  --pred-score-thr 0.12
```

21,31 对应的名词短语为 `watermelon`，45,59 对应的名词短语为 `a white bottle`。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/15080faf-048d-4201-a126-a9c773580f5e" width="70%"/>
</div>

**(4) 指代性表达式理解**

指代性表达式理解是指的用户输入一句语言描述，模型自动对其涉及到的指代性表达式进行理解, 不需要进行名词短语提取。

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

## 评测

我们所提供的评测脚本都是统一的，你只需要提前准备好数据，然后运行相关配置就可以了

(1) Zero-Shot COCO2017 val

```shell
# 单卡
python tools/test.py configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth

# 8 卡
./tools/dist_test.sh configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth 8
```

(2) Zero-Shot ODinW13

```shell
# 单卡
python tools/test.py configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw13.py \
        grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth

# 8 卡
./tools/dist_test.sh configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw13.py \
        grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth 8
```

## 评测数据集结果可视化

为了方便大家对模型预测结果进行可视化和分析，我们支持了评测数据集预测结果可视化，以指代性表达式理解为例用法如下：

```shell
python tools/test.py configs/mm_grounding_dino/refcoco/grounding_dino_swin-t_pretrain_zeroshot_refexp \
        grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth --work-dir refcoco_result --show-dir save_path
```

模型在推理过程中会将可视化结果保存到  `refcoco_result/{当前时间戳}/save_path` 路径下。其余评测数据集可视化只需要替换配置文件即可。

下面展示一些数据集的可视化结果： 左图为 GT，右图为预测结果

1. COCO2017 val 结果：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/3a0fa894-c0a5-4c1f-bdf0-1c6fd17abafa" width="70%"/>
</div>

2. Flickr30k Entities 结果：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/e9f2667f-9dca-464b-b995-599aa2731b34" width="70%"/>
</div>

3. DOD 结果：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/c71a306b-1055-4344-ba1d-ae4c57f2cb2f" width="70%"/>
</div>

4. RefCOCO val 结果：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/b175959d-d788-4b5e-8b11-e8e34753457f" width="70%"/>
</div>

5. RefCOCO testA 结果：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/c087f889-f96c-4355-8a15-7dc2738b4223" width="70%"/>
</div>

6. gRefCOCO val 结果：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/96c2e783-17da-462e-a7cf-937555e26c90" width="70%"/>
</div>

## 模型训练

如果想复现我们的结果，你可以在准备好数据集后，直接通过如下命令进行训练

```shell
# 单机 8 卡训练仅包括 obj365v1 数据集
./tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py 8
# 单机 8 卡训练包括 obj365v1/goldg/grit/v3det 数据集，其余数据集类似
./tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py 8
```

多机训练的用法请参考 [train.md](../../docs/zh_cn/user_guides/train.md)。MM-Grounding-DINO T 模型默认采用的是 32 张 3090Ti，如果你的总 bs 数不是 32x4=128，那么你需要手动的线性调整学习率。

### 预训练自定义格式说明

为了统一不同数据集的预训练格式，我们参考 [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino) 所设计的格式。具体来说分成 2 种格式

**(1) 目标检测数据格式 OD**

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

label字典中所对应的数值需要和相应的 label_map 一致。 instances 列表中的每一项都对应一个 bbox (x1y1x2y2 格式)。

**(2) phrase grounding 数据格式 VG**

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

tokens_positive 表示当前 phrase 在 caption 中的字符位置。

## 自定义数据集微调训练案例

为了方便用户针对自定义数据集进行下游微调，我们特意提供了以简单的 cat 数据集为例的微调训练案例。

### 1 数据准备

```shell
cd mmdetection
wget https://download.openmmlab.com/mmyolo/data/cat_dataset.zip
unzip cat_dataset.zip -d data/cat/
```

cat 数据集是一个单类别数据集，包含 144 张图片，已经转换为 coco 格式。

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205423220-c4b8f2fd-22ba-4937-8e47-1b3f6a8facd8.png" alt="cat dataset"/>
</div>

### 2 配置准备

由于 cat 数据集的简单性和数量较少，我们使用 8 卡训练 20 个 epoch，相应的缩放学习率，不训练语言模型，只训练视觉模型。

详细的配置信息可以在 [grounding_dino_swin-t_finetune_8xb4_20e_cat](grounding_dino_swin-t_finetune_8xb4_20e_cat.py) 中找到。

### 3 可视化和 Zero-Shot 评估

由于 MM Grounding DINO 是一个开放的检测模型，所以即使没有在 cat 数据集上训练，也可以进行检测和评估。

单张图片的可视化结果如下：

```shell
cd mmdetection
python demo/image_demo.py data/cat/images/IMG_20211205_120756.jpg configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth --texts cat.
```

测试集上的 Zero-Shot 评估结果如下：

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

### 4 模型训练

```shell
./tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py 8 --work-dir cat_work_dir
```

模型将会保存性能最佳的模型。在第 16 epoch 时候达到最佳，性能如下所示：

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

我们可以发现，经过微调训练后，cat 数据集的训练性能从 88.1 提升到了 90.1。同时由于数据集比较小，评估指标波动比较大。

## 模型自训练伪标签迭代生成和优化 pipeline

为了方便用户从头构建自己的数据集或者希望利用模型推理能力进行自举式伪标签迭代生成和优化，不断修改伪标签来提升模型性能，我们特意提供了相关的 pipeline。

由于我们定义了两种数据格式，为了演示我们也将分别进行说明。

### 1 目标检测格式

此处我们依然采用上述的 cat 数据集为例，假设我们目前只有一系列图片和预定义的类别，并不存在标注。

1. 生成初始 odvg 格式文件

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

会在 `data/cat` 目录下生成 `cat_train_od.json` 和 `cat_label_map.json` 两个文件。

2. 使用预训练模型进行推理，并保存结果

我们提供了直接可用的 [配置](grounding_dino_swin-t_pretrain_pseudo-labeling_cat.py), 如果你是其他数据集可以参考这个配置进行修改。

```shell
python tools/test.py configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_pseudo-labeling_cat.py \
    grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
```

会在 `data/cat` 目录下新生成 `cat_train_od_v1.json` 文件，你可以手动打开确认或者使用 [脚本](../../tools/analysis_tools/browse_grounding_raw.py) 可视化效果

```shell
python tools/analysis_tools/browse_grounding_raw.py data/cat/ cat_train_od_v1.json images --label-map-file cat_label_map.json -o your_output_dir --not-show
```

会在 your_output_dir 目录下生成可视化结果

3. 继续训练提高性能

在得到伪标签后，你可以混合一些预训练数据联合进行继续预训练，提升模型在当前数据集上的性能，然后重新运行 2 步骤，得到更准确的伪标签，如此循环迭代即可。

### 2 Phrase Grounding 格式

1. 生成初始 odvg 格式文件

Phrase Grounding 的自举流程要求初始时候提供每张图片对应的 caption 和提前切割好的 phrase 信息。以 flickr30k entities 图片为例，生成的典型的文件应该如下所示：

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

初始时候 bbox 必须要设置为 `[0, 0, 1, 1]`，因为这能确保程序正常运行，但是 bbox 的值并不会被使用。

```text
{"filename": "3028766968.jpg", "height": 375, "width": 500, "grounding": {"caption": "Man with a black shirt on sit behind a desk sorting threw a giant stack of people work with a smirk on his face .", "regions": [{"bbox": [0, 0, 1, 1], "phrase": "a giant stack of people", "tokens_positive": [[58, 81]]}, {"bbox": [0, 0, 1, 1], "phrase": "a black shirt", "tokens_positive": [[9, 22]]}, {"bbox": [0, 0, 1, 1], "phrase": "a desk", "tokens_positive": [[37, 43]]}, {"bbox": [0, 0, 1, 1], "phrase": "his face", "tokens_positive": [[103, 111]]}, {"bbox": [0, 0, 1, 1], "phrase": "Man", "tokens_positive": [[0, 3]]}]}}
{"filename": "6944134083.jpg", "height": 319, "width": 500, "grounding": {"caption": "Two men are competing in a horse race .", "regions": [{"bbox": [0, 0, 1, 1], "phrase": "Two men", "tokens_positive": [[0, 7]]}]}}
```

你可直接复制上面的文本，并假设将文本内容粘贴到命名为 `flickr_simple_train_vg.json` 文件中，并放置于提前准备好的 `data/flickr30k_entities` 数据集目录下，具体见数据准备文档。

2. 使用预训练模型进行推理，并保存结果

我们提供了直接可用的 [配置](grounding_dino_swin-t_pretrain_pseudo-labeling_flickr30k.py), 如果你是其他数据集可以参考这个配置进行修改。

```shell
python tools/test.py configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_pseudo-labeling_flickr30k.py \
    grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
```

会在 `data/flickr30k_entities` 目录下新生成 `flickr_simple_train_vg_v1.json` 文件，你可以手动打开确认或者使用 [脚本](../../tools/analysis_tools/browse_grounding_raw.py) 可视化效果

```shell
python tools/analysis_tools/browse_grounding_raw.py data/flickr30k_entities/ flickr_simple_train_vg_v1.json flickr30k_images -o your_output_dir --not-show
```

会在 `your_output_dir` 目录下生成可视化结果，如下图所示：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/a1c72d52-fa52-4ebe-b793-716d34e7b83f" width="50%"/>
</div>

3. 继续训练提高性能

在得到伪标签后，你可以混合一些预训练数据联合进行继续预训练，提升模型在当前数据集上的性能，然后重新运行 2 步骤，得到更准确的伪标签，如此循环迭代即可。
