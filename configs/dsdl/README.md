# DSDL: Standard Description Language for DataSet

## 1. Abstract

Data is the cornerstone of artificial intelligence. The efficiency of data acquisition, exchange, and application directly impacts the advances in technologies and applications. Over the long history of AI, a vast quantity of data sets have been developed and distributed. However, these datasets are defined in very different forms, which incurs significant overhead when it comes to exchange, integration, and utilization -- it is often the case that one needs to develop a new customized tool or script in order to incorporate a new dataset into a workflow.

To overcome such difficulties, we develop **Data Set Description Language (DSDL)**. More details please visit our [official documents](https://opendatalab.github.io/dsdl-docs/getting_started/overview/), dsdl datasets can be downloaded from our platform [OpenDataLab](https://opendatalab.com/).

## 2. Steps

- install dsdl:

  install by pip:

  ```
  pip install dsdl
  ```

  install by source code:

  ```
  git clone https://github.com/opendatalab/dsdl-sdk.git -b schema-dsdl
  cd dsdl-sdk
  python setup.py install
  ```

- install mmdet and pytorch:
  please refer this [installation documents](https://mmdetection.readthedocs.io/en/latest/get_started.html).

- train:

  - using single gpu:

  ```
  python tools/train.py {config_file}
  ```

  - using slrum:

  ```
  ./tools/slurm_train.sh {partition} {job_name} {config_file} {work_dir} {gpu_nums}
  ```

## 3. Test Results

- detection task:

  |  Datasets  |                                                                                         Model                                                                                          | box AP |           Config            |
  | :--------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----: | :-------------------------: |
  |  VOC07+12  |             [model](https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth)             | 80.3\* |   [config](./voc0712.py)    |
  |    COCO    |                   [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth)                   |  37.4  |     [config](./coco.py)     |
  | Objects365 |       [model](https://download.openmmlab.com/mmdetection/v2.0/objects365/faster_rcnn_r50_fpn_16x4_1x_obj365v2/faster_rcnn_r50_fpn_16x4_1x_obj365v2_20221220_175040-5910b015.pth)       |  19.8  | [config](./objects365v2.py) |
  | OpenImages | [model](https://download.openmmlab.com/mmdetection/v2.0/openimages/faster_rcnn_r50_fpn_32x2_cas_1x_openimages/faster_rcnn_r50_fpn_32x2_cas_1x_openimages_20220306_202424-98c630e5.pth) | 59.9\* | [config](./openimagesv6.py) |

  \*: box AP in voc metric and openimages metric, actually means AP_50.

- instance segmentation task:

  | Datasets |                                                                    Model                                                                     | box AP | mask AP |            Config            |
  | :------: | :------------------------------------------------------------------------------------------------------------------------------------------: | :----: | :-----: | :--------------------------: |
  |   COCO   | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth) |  38.1  |  34.7   | [config](./coco_instance.py) |
