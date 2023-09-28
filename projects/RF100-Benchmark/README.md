# Roboflow 100 Benchmark

> [Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark](https://arxiv.org/abs/2211.13523v3)

<!-- [Dataset] -->

## Abstract

The evaluation of object detection models is usually performed by optimizing a single metric, e.g. mAP, on a fixed set of datasets, e.g. Microsoft COCO and Pascal VOC. Due to image retrieval and annotation costs, these datasets consist largely of images found on the web and do not represent many real-life domains that are being modelled in practice, e.g. satellite, microscopic and gaming, making it difficult to assert the degree of generalization learned by the model. We introduce the Roboflow-100 (RF100) consisting of 100 datasets, 7 imagery domains, 224,714 images, and 805 class labels with over 11,170 labelling hours. We derived RF100 from over 90,000 public datasets, 60 million public images that are actively being assembled and labelled by computer vision practitioners in the open on the web application Roboflow Universe. By releasing RF100, we aim to provide a semantically diverse, multi-domain benchmark of datasets to help researchers test their model's generalizability with real-life data. RF100 download and benchmark replication are available on GitHub.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/71b0eb6f-d710-4100-9fb1-9d5485e07fdb"/>
</div>

## Code Structure

```text
# current path is projects/RF100-Benchmark/
├── configs
│         ├── dino_r50_fpn_ms_8xb8_tweeter-profile.py
│         ├── faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py
│         └── tood_r50_fpn_ms_8xb8_tweeter-profile.py
├── README.md
├── README_zh-CN.md
├── rf100
└── scripts
    ├── create_new_config.py # Based on the provided configuration, generate the training configuration of the remaining 99 datasets
    ├── datasets_links_640.txt # Dataset download link, from the official repo
    ├── download_dataset.py # Dataset download code, from the official repo
    ├── download_datasets.sh # Dataset download script, from the official repo
    ├── labels_names.json # Dataset information, from the official repo, but there are some errors so we modified it
    ├── parse_dataset_link.py # from the official repo
    ├── log_extract.py # Results collection and collation of training
    └── dist_train.sh # Training and evaluation startup script
    └── slurm_train.sh # Slurm Training and evaluation startup script
```

## Dataset Preparation

Roboflow 100 dataset is hosted by Roboflow platform, and detailed download scripts are provided in the [roboflow-100-benchmark](https://github.com/roboflow/roboflow-100-benchmark) repository. For simplicity, we use the official download script directly.

Before downloading the data, you need to register an account on the Roboflow platform to get the API key.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/6126e69e-85ce-4dec-8e7b-936c4fae29a6"/>
</div>

```shell
export ROBOFLOW_API_KEY = Your Private API Key
```

At the same time, you should also install the Roboflow package.

```shell
pip install roboflow
```

Finally, use the following command to download the dataset.

```shell
cd projects/RF100-Benchmark/
bash scripts/download_datasets.sh
```

Download the dataset, and a `rf100` folder will be generated in the current directory `projects/RF100-Benchmark/`, which contains all the datasets. The structure is as follows:

```text
# current path is projects/RF100-Benchmark/
├── README.md
├── README_zh-CN.md
└── scripts
    ├── datasets_links_640.txt
├── rf100
│    └── tweeter-profile
│    │    ├── train
|    |    |    ├── 0b3la49zec231_jpg.rf.8913f1b7db315c31d09b1d2f583fb521.jpg
|    |    |    ├──_annotations.coco.json
│    │    ├── valid
|    |    |    ├── 0fcjw3hbfdy41_jpg.rf.d61585a742f6e9d1a46645389b0073ff.jpg
|    |    |    ├──_annotations.coco.json
│    │    ├── test
|    |    |    ├── 0dh0to01eum41_jpg.rf.dcca24808bb396cdc07eda27a2cea2d4.jpg
|    |    |    ├──_annotations.coco.json
│    │    ├── README.dataset.txt
│    │    ├── README.roboflow.txt
│    └── 4-fold-defect
...
```

The dataset takes up a total of 12.3G of storage space. If you don't want to train and evaluate all models at once, you can modify the `scripts/datasets_links_640.txt` file and delete the links to the datasets you don't want to use.

Roboflow 100 dataset features are shown in the following figure

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/e2693662-3d16-49a4-af0b-2a03be7e16b6"/>
</div>

If you want to have a clear understanding of the dataset, you can check the [roboflow-100-benchmark](https://github.com/roboflow/roboflow-100-benchmark) repository, which provides many dataset analysis scripts.

## Model Training and Evaluation

If you want to train and evaluate all models at once, you can use the following command.

1. Single GPU Training

```shell
# current path is projects/RF100-Benchmark/
bash scripts/dist_train.sh configs/faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py 1
# Specify the save path
bash scripts/dist_train.sh configs/faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py 1 my_work_dirs
```

2. Distributed Multi-GPU Training

```shell
bash scripts/dist_train.sh configs/faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py 8
# Specify the save path
bash scripts/dist_train.sh configs/faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py 8 my_work_dirs
```

3. Slurm Training

```shell
bash scripts/slurm_train.sh configs/faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py 8
# Specify the save path
bash scripts/slurm_train.sh configs/faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py 8 my_work_dirs
```

After training, a `work_dirs` folder will be generated in the current directory, which contains the trained model weights and logs.

1. For the convenience of users to debug or only want to train specific datasets, we provide the `DEBUG` variable in `scripts/*_train.sh`, you only need to set it to 1, and specify the datasets you want to train in the `datasets_list` variable.
2. Considering that for various reasons, users may encounter training failures for certain datasets during the training process, we provide the `RETRY_PATH` variable, you only need to pass in the txt dataset list file, and the program will read the dataset in the file, and then only train specific datasets. If not provided, it is training the full dataset.

```shell
RETRY_PATH=failed_dataset_list.txt bash scripts/dist_train.sh configs/faster-rcnn_r50_fpn_ms_8xb8_tweeter-profile.py 8 my_work_dirs
```

The txt represents a dataset name on each line, as shown below (the blank line in the 4th line is indispensable):

```text
acl-x-ray
tweeter-profile
abdomen-mri

```

The txt file can also be generated using the `log_extract.py` script introduced later, without manually creating it.

## Model Summary

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/bb187af4-cdbf-40ba-8def-8870e239c8dd"/>
</div>

If you want to collect the results after the model is trained or during the training, you can execute the `log_extract.py` script, which will collect the information under `work_dirs` and output it in csv and xlsx format.

Before running the script, please make sure that `pandas` and `openpyxl` are installed

```shell
python scripts/log_extract.py faster_rcnn --epoch 25 --work-dirs my_work_dirs
```

- The first input parameter is used to generate the csv title, so you can enter any string, but it is recommended to enter the model name for easy viewing later.
- `--epoch` parameter refers to the number of model training epochs, which is used to parse the log. By default, we train 100 epochs for each dataset, but RepeatDataset is used in the configuration, so the actual training epoch is 25.
- `--work-dirs` is the working path where you save the trained model. The default is the `work_dirs` folder under the current path.

After running, the following three new files will be generated in `my_work_dirs`

```text
timestamp_detail.xlsx # Detailed information on the sorting of 100 datasets.
timestamp_sum.xlsx # Summary information of 100 datasets.
timestamp_eval.csv # Evaluation results of 100 datasets in the order of training.
failed_dataset_list.txt
```

Currently, we provide the evaluation results of the Faster RCNN, TOOD and DINO algorithms (no careful parameter tuning). You can also quickly evaluate your own model according to the above process.

Note:

1. Since there are a lot of 100 datasets, we cannot check each dataset, so if there is anything unreasonable, please feedback, we will fix it as soon as possible.
2. We also provide various scale summary results such as mAP_s, but because some data does not exist this scale bounding box, we ignore these datasets when summarizing.

## Custom Algorithm Benchmark

If users want to benchmark different algorithms for Roboflow 100, you only need to add algorithm configurations in the `projects/RF100-Benchmark/configs` folder.

Note: Since the internal running process is to replace the string in the user-provided configuration with the function of custom dataset, the configuration provided by the user must be the `tweeter-profile` dataset and must include the `data_root` and `class_name` variables, otherwise the program will report an error.

## Citation

```BibTeX
@misc{2211.13523,
Author = {Floriana Ciaglia and Francesco Saverio Zuppichini and Paul Guerrie and Mark McQuade and Jacob Solawetz},
Title = {Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark},
Year = {2022},
Eprint = {arXiv:2211.13523},
}
```
