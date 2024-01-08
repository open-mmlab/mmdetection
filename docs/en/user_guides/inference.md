# Inference with existing models

MMDetection provides hundreds of pre-trained detection models in [Model Zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html).
This note will show how to inference, which means using trained models to detect objects on images.

In MMDetection, a model is defined by a [configuration file](https://mmdetection.readthedocs.io/en/latest/user_guides/config.html) and existing model parameters are saved in a checkpoint file.

To start with, we recommend [RTMDet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) with this [configuration file](https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py) and this [checkpoint file](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth). It is recommended to download the checkpoint file to `checkpoints` directory.

## High-level APIs for inference - `Inferencer`

In OpenMMLab, all the inference operations are unified into a new interface - Inferencer. Inferencer is designed to expose a neat and simple API to users, and shares very similar interface across different OpenMMLab libraries.
A notebook demo can be found in [demo/inference_demo.ipynb](https://github.com/open-mmlab/mmdetection/blob/main/demo/inference_demo.ipynb).

### Basic Usage

You can get inference results for an image with only 3 lines of code.

```python
from mmdet.apis import DetInferencer

# Initialize the DetInferencer
inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco')

# Perform inference
inferencer('demo/demo.jpg', show=True)
```

The resulting output will be displayed in a new window:.

<div align="center">
    <img src='https://github.com/open-mmlab/mmdetection/assets/27466624/311df42d-640a-4a5b-9ad9-9ba7f3ec3a2f' />
</div>

```{note}
If you are running MMDetection on a server without GUI or via SSH tunnel with X11 forwarding disabled, the `show` option will not work. However, you can still save visualizations to files by setting `out_dir` arguments. Read [Dumping Results](#dumping-results) for details.
```

### Initialization

Each Inferencer must be initialized with a model. You can also choose the inference device during initialization.

#### Model Initialization

- To infer with MMDetection's pre-trained model, passing its name to the argument `model` can work. The weights will be automatically downloaded and loaded from OpenMMLab's model zoo.

  ```python
  inferencer = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco')
  ```

  There is a very easy to list all model names in MMDetection.

  ```python
  # models is a list of model names, and them will print automatically
  models = DetInferencer.list_models('mmdet')
  ```

  You can load another weight by passing its path/url to `weights`.

  ```python
  inferencer = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco', weights='path/to/rtmdet.pth')
  ```

- To load custom config and weight, you can pass the path to the config file to `model` and the path to the weight to `weights`.

  ```python
  inferencer = DetInferencer(model='path/to/rtmdet_config.py', weights='path/to/rtmdet.pth')
  ```

- By default, [MMEngine](https://github.com/open-mmlab/mmengine/) dumps config to the weight. If you have a weight trained on MMEngine, you can also pass the path to the weight file to `weights` without specifying `model`:

  ```python
  # It will raise an error if the config file cannot be found in the weight. Currently, within the MMDetection model repository, only the weights of ddq-detr-4scale_r50 can be loaded in this manner.
  inferencer = DetInferencer(weights='https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq-detr-4scale_r50_8xb2-12e_coco/ddq-detr-4scale_r50_8xb2-12e_coco_20230809_170711-42528127.pth')
  ```

- Passing config file to `model` without specifying `weight` will result in a randomly initialized model.

### Device

Each Inferencer instance is bound to a device.
By default, the best device is automatically decided by [MMEngine](https://github.com/open-mmlab/mmengine/). You can also alter the device by specifying the `device` argument. For example, you can use the following code to create an Inferencer on GPU 1.

```python
inferencer = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco', device='cuda:1')
```

To create an Inferencer on CPU:

```python
inferencer = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco', device='cpu')
```

Refer to [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) for all the supported forms.

### Inference

Once the Inferencer is initialized, you can directly pass in the raw data to be inferred and get the inference results from return values.

#### Input

Input can be either of these types:

- str: Path/URL to the image.

  ```python
  inferencer('demo/demo.jpg')
  ```

- array: Image in numpy array. It should be in BGR order.

  ```python
  import mmcv
  array = mmcv.imread('demo/demo.jpg')
  inferencer(array)
  ```

- list: A list of basic types above. Each element in the list will be processed separately.

  ```python
  inferencer(['img_1.jpg', 'img_2.jpg])
  # You can even mix the types
  inferencer(['img_1.jpg', array])
  ```

- str: Path to the directory. All images in the directory will be processed.

  ```python
  inferencer('path/to/your_imgs/')
  ```

### Output

By default, each `Inferencer` returns the prediction results in a dictionary format.

- `visualization` contains the visualized predictions.

- `predictions` contains the predictions results in a json-serializable format. But it's an empty list by default unless `return_vis=True`.

```python
{
      'predictions' : [
        # Each instance corresponds to an input image
        {
          'labels': [...],  # int list of length (N, )
          'scores': [...],  # float list of length (N, )
          'bboxes': [...],  # 2d list of shape (N, 4), format: [min_x, min_y, max_x, max_y]
        },
        ...
      ],
      'visualization' : [
        array(..., dtype=uint8),
      ]
  }
```

If you wish to get the raw outputs from the model, you can set `return_datasamples` to `True` to get the original [DataSample](advanced_guides/structures.md), which will be stored in `predictions`.

#### Dumping Results

Apart from obtaining predictions from the return value, you can also export the predictions/visualizations to files by setting `out_dir` and `no_save_pred`/`no_save_vis` arguments.

```python
inferencer('demo/demo.jpg', out_dir='outputs/', no_save_pred=False)
```

Results in the directory structure like:

```text
outputs
├── preds
│   └── demo.json
└── vis
    └── demo.jpg
```

The filename of each file is the same as the corresponding input image filename. If the input image is an array, the filename will be a number starting from 0.

#### Batch Inference

You can customize the batch size by setting `batch_size`. The default batch size is 1.

### API

Here are extensive lists of parameters that you can use.

- **DetInferencer.\_\_init\_\_():**

| Arguments       | Type          | Type    | Description                                                                                                                                                                                                                                                                                              |
| --------------- | ------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`         | str, optional | None    | Path to the config file or the model name defined in metafile. For example, it could be 'rtmdet-s' or 'rtmdet_s_8xb32-300e_coco' or 'configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'. If the model is not specified, the user must provide the `weights` saved by MMEngine which contains the config string. |
| `weights`       | str, optional | None    | Path to the checkpoint. If it is not specified and `model` is a model name of metafile, the weights will be loaded from metafile.                                                                                                                                                                        |
| `device`        | str, optional | None    | Device used for inference, accepting all allowed strings by `torch.device`. E.g., 'cuda:0' or 'cpu'. If None, the available device will be automatically used.                                                                                                                                           |
| `scope`         | str, optional | 'mmdet' | The scope of the model.                                                                                                                                                                                                                                                                                  |
| `palette`       | str           | 'none'  | Color palette used for visualization. The order of priority is palette -> config -> checkpoint.                                                                                                                                                                                                          |
| `show_progress` | bool          | True    | Control whether to display the progress bar during the inference process.                                                                                                                                                                                                                                |

- **DetInferencer.\_\_call\_\_()**

| Arguments            | Type                      | Default      | Description                                                                                                                                                                                                                                                    |
| -------------------- | ------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `inputs`             | str/list/tuple/np.array   | **required** | It can be a path to an image/a folder, an np array or a list/tuple (with img paths or np arrays)                                                                                                                                                               |
| `batch_size`         | int                       | 1            | Inference batch size.                                                                                                                                                                                                                                          |
| `print_result`       | bool                      | False        | Whether to print the inference result to the console.                                                                                                                                                                                                          |
| `show`               | bool                      | False        | Whether to display the visualization results in a popup window.                                                                                                                                                                                                |
| `wait_time`          | float                     | 0            | The interval of show(s).                                                                                                                                                                                                                                       |
| `no_save_vis`        | bool                      | False        | Whether to force not to save prediction vis results.                                                                                                                                                                                                           |
| `draw_pred`          | bool                      | True         | Whether to draw predicted bounding boxes.                                                                                                                                                                                                                      |
| `pred_score_thr`     | float                     | 0.3          | Minimum score of bboxes to draw.                                                                                                                                                                                                                               |
| `return_datasamples` | bool                      | False        | Whether to return results as DataSamples. If False, the results will be packed into a dict.                                                                                                                                                                    |
| `print_result`       | bool                      | False        | Whether to print the inference result to the console.                                                                                                                                                                                                          |
| `no_save_pred`       | bool                      | True         | Whether to force not to save prediction results.                                                                                                                                                                                                               |
| `out_dir`            | str                       | ''           | Output directory of results.                                                                                                                                                                                                                                   |
| `texts`              | str/list\[str\], optional | None         | Text prompts.                                                                                                                                                                                                                                                  |
| `stuff_texts`        | str/list\[str\], optional | None         | Stuff text prompts of open panoptic task.                                                                                                                                                                                                                      |
| `custom_entities`    | bool                      | False        | Whether to use custom entities. Only used in GLIP.                                                                                                                                                                                                             |
| \*\*kwargs           |                           |              | Other keyword arguments passed to :meth:`preprocess`, :meth:`forward`, :meth:`visualize` and :meth:`postprocess`. Each key in kwargs should be in the corresponding set of `preprocess_kwargs`, `forward_kwargs`, `visualize_kwargs` and `postprocess_kwargs`. |

## Demos

We also provide four demo scripts, implemented with high-level APIs and supporting functionality codes.
Source codes are available [here](https://github.com/open-mmlab/mmdetection/blob/main/demo).

### Image demo

This script performs inference on a single image.

```shell
python demo/image_demo.py \
    ${IMAGE_FILE} \
    ${CONFIG_FILE} \
    [--weights ${WEIGHTS}] \
    [--device ${GPU_ID}] \
    [--pred-score-thr ${SCORE_THR}]
```

Examples:

```shell
python demo/image_demo.py demo/demo.jpg \
    configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    --weights checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
    --device cpu
```

### Webcam demo

This is a live demo from a webcam.

```shell
python demo/webcam_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--camera-id ${CAMERA-ID}] \
    [--score-thr ${SCORE_THR}]
```

Examples:

```shell
python demo/webcam_demo.py \
    configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth
```

### Video demo

This script performs inference on a video.

```shell
python demo/video_demo.py \
    ${VIDEO_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--score-thr ${SCORE_THR}] \
    [--out ${OUT_FILE}] \
    [--show] \
    [--wait-time ${WAIT_TIME}]
```

Examples:

```shell
python demo/video_demo.py demo/demo.mp4 \
    configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
    --out result.mp4
```

#### Video demo with GPU acceleration

This script performs inference on a video with GPU acceleration.

```shell
python demo/video_gpuaccel_demo.py \
    ${VIDEO_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--score-thr ${SCORE_THR}] \
    [--nvdecode] \
    [--out ${OUT_FILE}] \
    [--show] \
    [--wait-time ${WAIT_TIME}]
```

Examples:

```shell
python demo/video_gpuaccel_demo.py demo/demo.mp4 \
    configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
    --nvdecode --out result.mp4
```

### Large-image inference demo

This is a script for slicing inference on large images.

```
python demo/large_image_demo.py \
	${IMG_PATH} \
	${CONFIG_FILE} \
	${CHECKPOINT_FILE} \
	--device ${GPU_ID}  \
	--show \
	--tta  \
	--score-thr ${SCORE_THR} \
	--patch-size ${PATCH_SIZE} \
	--patch-overlap-ratio ${PATCH_OVERLAP_RATIO} \
	--merge-iou-thr ${MERGE_IOU_THR} \
	--merge-nms-type ${MERGE_NMS_TYPE} \
	--batch-size ${BATCH_SIZE} \
	--debug \
	--save-patch
```

Examples:

```shell
# inferecnce without tta
wget -P checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth

python demo/large_image_demo.py \
    demo/large_image.jpg \
    configs/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py \
    checkpoint/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth

# inference with tta
wget -P checkpoint https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth

python demo/large_image_demo.py \
    demo/large_image.jpg \
    configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    checkpoint/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --tta

```

## Multi-modal algorithm inference demo and evaluation

As multimodal vision algorithms continue to evolve, MMDetection has also supported such algorithms. This section demonstrates how to use the demo and eval scripts corresponding to multimodal algorithms using the GLIP algorithm and model as the example. Moreover, MMDetection integrated a [gradio_demo project](../../../projects/gradio_demo/), which allows developers to quickly play with all image input tasks in MMDetection on their local devices. Check the [document](../../../projects/gradio_demo/README.md) for more details.

### Preparation

Please first make sure that you have the correct dependencies installed:

```shell
# if source
pip install -r requirements/multimodal.txt

# if wheel
mim install mmdet[multimodal]
```

MMDetection has already implemented GLIP algorithms and provided the weights, you can download directly from urls:

```shell
cd mmdetection
wget https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth
```

### Inference

Once the model is successfully downloaded, you can use the `demo/image_demo.py` script to run the inference.

```shell
python demo/image_demo.py demo/demo.jpg glip_tiny_a_mmdet-b3654169.pth --texts bench
```

Demo result will be similar to this:

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/234547841-266476c8-f987-4832-8642-34357be621c6.png" height="300"/>
</div>

If users would like to detect multiple targets, please declare them in the format of `xx. xx` after the `--texts`.

```shell
python demo/image_demo.py demo/demo.jpg glip_tiny_a_mmdet-b3654169.pth --texts 'bench. car'
```

And the result will be like this one:

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/234548156-ef9bbc2e-7605-4867-abe6-048b8578893d.png" height="300"/>
</div>

You can also use a sentence as the input prompt for the `--texts` field, for example:

```shell
python demo/image_demo.py demo/demo.jpg glip_tiny_a_mmdet-b3654169.pth --texts 'There are a lot of cars here.'
```

The result will be similar to this:

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/234548490-d2e0a16d-1aad-4708-aea0-c829634fabbd.png" height="300"/>
</div>

### Evaluation

The GLIP implementation in MMDetection does not have any performance degradation, our benchmark is as follows:

| Model                   | official mAP | mmdet mAP |
| ----------------------- | :----------: | :-------: |
| glip_A_Swin_T_O365.yaml |     42.9     |   43.0    |
| glip_Swin_T_O365.yaml   |     44.9     |   44.9    |
| glip_Swin_L.yaml        |     51.4     |   51.3    |

Users can use the test script we provided to run evaluation as well. Here is a basic example:

```shell
# 1 gpu
python tools/test.py configs/glip/glip_atss_swin-t_fpn_dyhead_pretrain_obj365.py glip_tiny_a_mmdet-b3654169.pth

# 8 GPU
./tools/dist_test.sh configs/glip/glip_atss_swin-t_fpn_dyhead_pretrain_obj365.py glip_tiny_a_mmdet-b3654169.pth 8
```
