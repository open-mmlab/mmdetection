# Open-vocabulary COCO

## Data preparation

Prepare data following [MMDetection](https://github.com/open-mmlab/mmdetection).
Obtain the json files for OV-COCO from [GoogleDrive](https://drive.google.com/drive/folders/1O6rt6WN2ePPg6j-wVgF89T7ql2HiuRIG?usp=sharing) and put them
under `data/coco/wusize`
The data structure looks like:

```text
checkpoints/
├── clip_vitb32.pth
├── res50_fpn_soco_star_400.pth
data/
├── coco
│   ├── annotations
│   │   ├── instances_{train,val}2017.json
│   ├── wusize
│   │   ├── instances_train2017_base.json
│   │   ├── instances_val2017_base.json
│   │   ├── instances_val2017_novel.json
│   │   ├── captions_train2017_tags_allcaps.json
│   ├── train2017
│   ├── val2017
│   ├── test2017
```

Otherwise, generate the json files using the following scripts

```bash
python tools/pre_processors/keep_coco_base.py \
      --json_path data/coco/annotations/instances_train2017.json \
      --out_path data/coco/wusize/instances_train2017_base.json
```

```bash
python tools/pre_processors/keep_coco_base.py \
      --json_path data/coco/annotations/instances_val2017.json \
      --out_path data/coco/wusize/instances_val2017_base.json
```

```bash
python tools/pre_processors/keep_coco_novel.py \
      --json_path data/coco/annotations/instances_val2017.json \
      --out_path data/coco/wusize/instances_val2017_novel.json
```

The json file for caption supervision `captions_train2017_tags_allcaps.json` is obtained following
[Detic](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md#:~:text=Next%2C%20we%20preprocess%20the%20COCO%20caption%20data%3A). Put it under
`data/coco/wusize`.

### Class Embeddings

As the training on COCO tends to converge to base categories, we use the output of the last attention
layer for classification. Generate the class embeddings by

```bash
python tools/hand_craft_prompt.py --model_version ViT-B/32 --ann data/coco/annotations/instances_val2017.json \
--out_path data/metadata/coco_clip_hand_craft.npy --dataset coco
```

The generated file `data/metadata/coco_clip_hand_craft_attn12.npy` is used for training and testing.

## Testing

### Open Vocabulary COCO

The implementation based on MMDet3.x achieves better results compared to the results reported in the paper.

|           | Backbone | Method | Supervision  | Novel AP50 |                         Config                          |                                                                                           Download                                                                                            |
| :-------: | :------: | :----: | :----------: | :--------: | :-----------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   Paper   | R-50-FPN | BARON  |     CLIP     |    34.0    |                            -                            |                                                                                               -                                                                                               |
| This Repo | R-50-FPN | BARON  |     CLIP     |    34.6    | [config](baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py)  | [model](https://drive.google.com/drive/folders/1JTM0uoPQZtq7lnhZxCBwjxBUca9omYR9?usp=sharing) \|  [log](https://drive.google.com/drive/folders/1JTM0uoPQZtq7lnhZxCBwjxBUca9omYR9?usp=sharing) |
|   Paper   | R-50-C4  | BARON  | COCO Caption |    33.1    |                            -                            |                                                                                               -                                                                                               |
| This Repo | R-50-C4  | BARON  | COCO Caption |    35.1    | [config](baron_caption_faster_rcnn_r50_caffe_c4_90k.py) | [model](https://drive.google.com/drive/folders/1b-ueEz57alju9qamADm7BmDCaL-NWnSn?usp=sharing) \|  [log](https://drive.google.com/drive/folders/1b-ueEz57alju9qamADm7BmDCaL-NWnSn?usp=sharing) |
| This Repo | R-50-C4  | BARON  |     CLIP     |    34.0    |   [config](baron_kd_faster_rcnn_r50_caffe_c4_90k.py)    | [model](https://drive.google.com/drive/folders/1ckS8Cju2xQyHfxMsQRPd5h7qKhwlWOyV?usp=sharing) \|  [log](https://drive.google.com/drive/folders/1ckS8Cju2xQyHfxMsQRPd5h7qKhwlWOyV?usp=sharing) |

To test the models, run

```bash
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=12 bash tools/slurm_test.sh PARTITION test \
path/to/the/cfg/file path/to/the/checkpoint
```

## Training

### Knowledge Distillation on CLIP

Train the detector based on FasterRCNN+ResNet50+FPN with SyncBN and SOCO pre-trained model. Obtain the SOCO pre-trained
model from [GoogleDrive](https://drive.google.com/file/d/1rIW9IXjWEnFZa4klZuZ5WNSchRYaOC0x/view?usp=sharing) and put it
under `checkpoints`.

```bash
GPUS=16 GPUS_PER_NODE=8 CPUS_PER_TASK=12 bash tools/slurm_train.sh PARTITION train \
configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
path/to/save/logs/and/checkpoints
```

We can also train a detector based on FasterRCNN+ResNet50C4.

```bash
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=12 bash tools/slurm_train.sh PARTITION train \
configs/baron/ov_coco/baron_kd_faster_rcnn_r50_c4_90k.py \
path/to/save/logs/and/checkpoints
```

### Caption Supervision

Train the detector based on FasterRCNN+ResNet50C4.

```bash
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=12 bash tools/slurm_train.sh PARTITION train \
configs/baron/ov_coco/baron_caption_faster_rcnn_r50_caffe_c4_90k.py \
path/to/save/logs/and/checkpoints
```
