# Model Deployment

- [Installation](#installation)
  - [Install mmdet](#install-mmdet)
  - [Install mmdeploy](#install-mmdeploy)
- [Convert model](#convert-model)
- [Model specification](#model-specification)
- [Model inference](#model-inference)
  - [Backend model inference](#backend-model-inference)
  - [SDK model inference](#sdk-model-inference)
- [Supported models](#supported-models)

______________________________________________________________________

The deployment of OpenMMLab codebases, including MMDetection, MMClassification and so on are supported by [MMDeploy](https://github.com/open-mmlab/mmdeploy).
The latest deployment guide for MMDetection can be found from [here](https://mmdeploy.readthedocs.io/en/dev-1.x/04-supported-codebases/mmdet.html).

## Installation

### Install mmdet

Please follow the [installation guide](https://mmdetection.readthedocs.io/en/3.x/get_started.html) to install mmdet.

### Install mmdeploy

There are several methods to install mmdeploy, among which you can choose an appropriate one according to your target platform and device.

**Method I:** Install precompiled package

> **TODO**. MMDeploy hasn't released based on 1.x branch.

**Method II:** Build using scripts

If your target platform is **Ubuntu 18.04 or later version**, we encourage you to run
[scripts](https://mmdeploy.readthedocs.io/en/dev-1.x/01-how-to-build/build_from_script.html). For example, the following commands install mmdeploy as well as inference engine - `ONNX Runtime`.

```shell
git clone --recursive -b dev-1.x https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
```

**Method III:** Build from source

If neither **I** nor **II** meets your requirements, [building mmdeploy from source](https://mmdeploy.readthedocs.io/en/dev-1.x/01-how-to-build/build_from_source.html) is the last option.

## Convert model

You can use [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/tools/deploy.py) to convert mmdet models to the specified backend models. Its detailed usage can be learned from [here](../02-how-to-run/convert_model.md).

The command below shows an example about converting `Faster R-CNN` model to onnx model that can be inferred by ONNX Runtime.

```shell
cd mmdeploy
# download faster r-cnn model from mmdet model zoo
mim download mmdet --config faster-rcnn_r50_fpn_1x_coco --dest .
# convert mmdet model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmdet/detection/detection_onnxruntime_dynamic.py \
    faster-rcnn_r50_fpn_1x_coco.py \
    faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    demo/resources/det.jpg \
    --work-dir mmdeploy_models/mmdet/ort \
    --device cpu \
    --show \
    --dump-info
```

It is crucial to specify the correct deployment config during model conversion. We've already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/dev-1.x/configs/mmdet) of all supported backends for mmdetection, under which the config file path follows the pattern:

```
{task}/{task}_{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{task}:** task in mmdetection.

  There are two of them. One is `detection` and the other is `instance-seg`, indicating instance segmentation.

  mmdet models like `RetinaNet`, `Faster R-CNN` and `DETR` and so on belongs to `detection` task. While `Mask R-CNN` is one of `instance-seg` models. You can find more of them in chapter [Supported models](#supported-models).

  **DO REMEMBER TO USE** `detection/detection_*.py` deployment config file when trying to convert detection models and use `instance-seg/instance-seg_*.py` to deploy instance segmentation models.

- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml etc.

- **{precision}:** fp16, int8. When it's empty, it means fp32

- **{static | dynamic}:** static shape or dynamic shape

- **{shape}:** input shape or shape range of a model

Therefore, in the above example, you can also convert `faster r-cnn` to other backend models by changing the deployment config file `detection_onnxruntime_dynamic.py` to [others](https://github.com/open-mmlab/mmdeploy/tree/dev-1.x/configs/mmdet/detection), e.g., converting to tensorrt-fp16 model by `detection_tensorrt-fp16_dynamic-320x320-1344x1344.py`.

```{tip}
When converting mmdet models to tensorrt models, --device should be set to "cuda"
```

## Model specification

Before moving on to model inference chapter, let's know more about the converted model structure which is very important for model inference.

The converted model locates in the working directory like `mmdeploy_models/mmdet/ort` in the previous example. It includes:

```
mmdeploy_models/mmdet/ort
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- \***.json**: the necessary information for mmdeploy SDK

The whole package **mmdeploy_models/mmdet/ort** is defined as **mmdeploy SDK model**, i.e., **mmdeploy SDK model** includes both backend model and inference meta information.

## Model inference

### Backend model inference

Take the previous converted `end2end.onnx` model as an example, you can use the following code to inference the model and visualize the results.

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/mmdet/detection/detection_onnxruntime_dynamic.py'
model_cfg = './faster-rcnn_r50_fpn_1x_coco.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmdet/ort/end2end.onnx']
image = './demo/resources/det.jpg'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='output_detection.png')
```

### SDK model inference

You can also perform SDK model inference like following,

```python
from mmdeploy_python import Detector
import cv2

img = cv2.imread('./demo/resources/det.jpg')

# create a detector
detector = Detector(model_path='./mmdeploy_models/mmdet/ort', device_name='cpu', device_id=0)
# perform inference
bboxes, labels, masks = detector(img)

# visualize inference result
indices = [i for i in range(len(bboxes))]
for index, bbox, label_id in zip(indices, bboxes, labels):
  [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
  if score < 0.3:
    continue

  cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

cv2.imwrite('output_detection.png', img)
```

Besides python API, mmdeploy SDK also provides other FFI (Foreign Function Interface), such as C, C++, C#, Java and so on. You can learn their usage from [demos](https://github.com/open-mmlab/mmdeploy/tree/dev-1.x/demo).

## Supported models

Please refer to [here](https://mmdeploy.readthedocs.io/en/dev-1.x/04-supported-codebases/mmdet.html#supported-models) for the supported model list.
