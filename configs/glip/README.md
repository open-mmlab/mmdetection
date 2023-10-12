# GLIP: Grounded Language-Image Pre-training

> [GLIP: Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857)

<!-- [ALGORITHM] -->

## Abstract

This paper presents a grounded language-image pre-training (GLIP) model for learning object-level, language-aware, and semantic-rich visual representations. GLIP unifies object detection and phrase grounding for pre-training. The unification brings two benefits: 1) it allows GLIP to learn from both detection and grounding data to improve both tasks and bootstrap a good grounding model; 2) GLIP can leverage massive image-text pairs by generating grounding boxes in a self-training fashion, making the learned representation semantic-rich. In our experiments, we pre-train GLIP on 27M grounding data, including 3M human-annotated and 24M web-crawled image-text pairs. The learned representations demonstrate strong zero-shot and few-shot transferability to various object-level recognition tasks. 1) When directly evaluated on COCO and LVIS (without seeing any images in COCO during pre-training), GLIP achieves 49.8 AP and 26.9 AP, respectively, surpassing many supervised baselines. 2) After fine-tuned on COCO, GLIP achieves 60.8 AP on val and 61.5 AP on test-dev, surpassing prior SoTA. 3) When transferred to 13 downstream object detection tasks, a 1-shot GLIP rivals with a fully-supervised Dynamic Head.

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/b87228d7-f000-4a5d-b103-fe535984417a"/>
</div>

## Installation

```shell
cd $MMDETROOT

# source installation
pip install -r requirements/multimodal.txt

# or mim installation
mim install mmdet[multimodal]
```

```shell
cd $MMDETROOT

wget https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth

python demo/image_demo.py demo/demo.jpg \
configs/glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py \
--weights glip_tiny_a_mmdet-b3654169.pth \
--texts 'bench. car'
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/7b450d96-81ac-462a-92bc-0d4ae7b8721c" width="40%"/>
</div>

## NOTE

GLIP utilizes BERT as the language model, which requires access to https://huggingface.co/. If you encounter connection errors due to network access, you can download the required files on a computer with internet access and save them locally. Finally, modify the `lang_model_name` field in the config to the local path. Please refer to the following code:

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

## Results and Models

|   Model    | Zero-shot or Finetune | COCO mAP | Official COCO mAP |       Pre-Train Data       |                                 Config                                  |                                                                                                                                                                                                   Download                                                                                                                                                                                                    |
| :--------: | :-------------------: | :------: | ----------------: | :------------------------: | :---------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| GLIP-T (A) |       Zero-shot       |   43.0   |              42.9 |            O365            |       [config](glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py)        |                                                                                                                                                         [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth)                                                                                                                                                          |
| GLIP-T (A) |       Finetune        |   53.3   |              52.9 |            O365            |   [config](glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)   | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_180419-e6addd96.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_180419.log.json) |
| GLIP-T (B) |       Zero-shot       |   44.9   |              44.9 |            O365            |       [config](glip_atss_swin-t_b_fpn_dyhead_pretrain_obj365.py)        |                                                                                                                                                         [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_b_mmdet-6dfbd102.pth)                                                                                                                                                          |
| GLIP-T (B) |       Finetune        |   54.1   |              53.8 |            O365            |   [config](glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)   | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230916_163538-650323ba.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230916_163538.log.json) |
| GLIP-T (C) |       Zero-shot       |   46.7   |              46.7 |         O365,GoldG         |    [config](glip_atss_swin-t_c_fpn_dyhead_pretrain_obj365-goldg.py)     |                                                                                                                                                         [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_c_mmdet-2fc427dd.pth)                                                                                                                                                          |
| GLIP-T (C) |       Finetune        |   55.2   |              55.1 |         O365,GoldG         |   [config](glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)   | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_182935-4ba3fc3b.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_182935.log.json) |
|   GLIP-T   |       Zero-shot       |   46.6   |              46.6 |    O365,GoldG,CC3M,SBU     | [config](glip_atss_swin-t_fpn_dyhead_pretrain_obj365-goldg-cc3m-sub.py) |                                                                                                                                                          [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_mmdet-c24ce662.pth)                                                                                                                                                           |
|   GLIP-T   |       Finetune        |   55.4   |              55.2 |    O365,GoldG,CC3M,SBU     |    [config](glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)    |     [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_224410-ba97be24.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_224410.log.json)     |
|   GLIP-L   |       Zero-shot       |   51.3   |              51.4 | FourODs,GoldG,CC3M+12M,SBU |       [config](glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py)       |                                                                                                                                                            [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_l_mmdet-abfe026b.pth)                                                                                                                                                            |
|   GLIP-L   |       Finetune        |   59.4   |                   | FourODs,GoldG,CC3M+12M,SBU |    [config](glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)    |     [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230910_100800-e9be4274.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230910_100800.log.json)     |

Note:

1. The weights corresponding to the zero-shot model are adopted from the official weights and converted using the [script](../../tools/model_converters/glip_to_mmdet.py). We have not retrained the model for the time being.
2. Finetune refers to fine-tuning on the COCO 2017 dataset. The L model is trained using 16 A100 GPUs, while the remaining models are trained using 16 NVIDIA GeForce 3090 GPUs.
3. Taking the GLIP-T(A) model as an example, I trained it twice using the official code, and the fine-tuning mAP were 52.5 and 52.6. Therefore, the mAP we achieved in our reproduction is higher than the official results. The main reason is that we modified the `weight_decay` parameter.
4. Our experiments revealed that training for 24 epochs leads to overfitting. Therefore, we chose the best-performing model. If users want to train on a custom dataset, it is advisable to shorten the number of epochs and save the best-performing model.
5. Due to the official absence of fine-tuning hyperparameters for the GLIP-L model, we have not yet reproduced the official accuracy. I have found that overfitting can also occur, so it may be necessary to consider custom modifications to data augmentation and model enhancement. Given the high cost of training, we have not conducted any research on this matter at the moment.
