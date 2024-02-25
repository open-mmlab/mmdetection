# Prompt Tuning For mmGroundingDINO

## Description

Implement prompt tuning feature on Grounded DINO. This fine-tuning method does not change the pre-train weight, and the trained prompt and text prompt can be used together. Pre-train weight in [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino).

## Training data

Data format: COCO
Category id in annotation starts from 0

## Special config parameters

In the `prompt_cfg` section of the config file
`class_num` must match the total number of categories in the training annotation.
`prompt_length` represents how many vectors are used to represent a single category, which has a significant impact on accuracy.

More details in grounding_dino_swinT_prompt.py

### Training

```bash
bash ./train/dist_train.sh config/grounding_dino_swinT_prompt.py {NUM_GPUS}
```

### After Training

After training, it is necessary to separate the learnable prompt from the saved model weight.

```bash
python model_split_to_prompt_pth.py --weight_path {saved_model_path} --real_name_list {name1,name2,...}  --save_path {save_dir_path}
```

real_name_list: used to display the class name during visualization

### Visualization

```bash
python single_image_inference.py --inputs {image path} --model {config path} --texts {text description} --prompt_pth {prompt path}
```

test_pipeline's PackDetInputs in config file must include 'prompt_pth'.
prompt_pth and texts can be used simultaneously.

## Citation

This feature is referenced for the following work.

```BibTeX
@article{chen2023exploration,
  title={Exploration of visual prompt in Grounded pre-trained open-set detection},
  author={Chen, Qibo and Jin, Weizhong and Li, Shuchang and Liu, Mengdi and Yu, Li and Jiang, Jian and Wang, Xiaozheng},
  journal={arXiv preprint arXiv:2312.08839},
  year={2023}
}
```

```BibTeX
@inproceedings{li2022grounded,
  title={Grounded language-image pre-training},
  author={Li, Liunian Harold and Zhang, Pengchuan and Zhang, Haotian and Yang, Jianwei and Li, Chunyuan and Zhong, Yiwu and Wang, Lijuan and Yuan, Lu and Zhang, Lei and Hwang, Jenq-Neng and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10965--10975},
  year={2022}
}
```
