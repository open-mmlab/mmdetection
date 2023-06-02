# Model Deployment

The deployment of OpenMMLab codebases, including MMDetection, MMClassification and so on are supported by [MMDeploy](https://github.com/open-mmlab/mmdeploy).
The latest deployment guide for MMDetection can be found from [here](https://mmdeploy.readthedocs.io/en/dev-1.x/04-supported-codebases/mmdet.html).

This tutorial is organized as follows:

- [Installation](#installation)
- [Convert model](#convert-model)
- [Model specification](#model-specification)
- [Model inference](#model-inference)
  - [Backend model inference](#backend-model-inference)
  - [SDK model inference](#sdk-model-inference)
- [Supported models](#supported-models)

## Installation

Please follow the [guide](https://mmdetection.readthedocs.io/en/latest/get_started.html) to install mmdet. And then install mmdeploy from source by following [this](https://mmdeploy.readthedocs.io/en/1.x/get_started.html#installation) guide.

```{note}
If you install mmdeploy prebuilt package, please also clone its repository by 'git clone https://github.com/open-mmlab/mmdeploy.git --depth=1' to get the deployment config files.
```

## Convert model

Suppose mmdetection and mmdeploy repositories are in the same directory, and the working directory is the root path of mmdetection.

Take [Faster R-CNN](https://github.com/open-mmlab/mmdetection/blob/main/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py) model as an example. You can download its checkpoint from [here](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth), and then convert it to onnx model as follows:

```python
from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK

img = 'demo/demo.jpg'
work_dir = 'mmdeploy_models/mmdet/onnx'
save_file = 'end2end.onnx'
deploy_cfg = '../mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py'
model_cfg = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
model_checkpoint = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cpu'

# 1. convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
           model_checkpoint, device)

# 2. extract pipeline info for inference by MMDeploy SDK
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint,
           device=device)
```

It is crucial to specify the correct deployment config during model conversion. MMDeploy has already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/1.x/configs/mmdet) of all supported backends for mmdetection, under which the config file path follows the pattern:

```
{task}/{task}_{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{task}:** task in mmdetection.

  There are two of them. One is `detection` and the other is `instance-seg`, indicating instance segmentation.

  mmdet models like `RetinaNet`, `Faster R-CNN` and `DETR` and so on belongs to `detection` task. While `Mask R-CNN` is one of `instance-seg` models.

  **DO REMEMBER TO USE** `detection/detection_*.py` deployment config file when trying to convert detection models and use `instance-seg/instance-seg_*.py` to deploy instance segmentation models.

- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml etc.

- **{precision}:** fp16, int8. When it's empty, it means fp32

- **{static | dynamic}:** static shape or dynamic shape

- **{shape}:** input shape or shape range of a model

Therefore, in the above example, you can also convert `Faster R-CNN` to tensorrt-fp16 model by `detection_tensorrt-fp16_dynamic-320x320-1344x1344.py`.

```{tip}
When converting mmdet models to tensorrt models, --device should be set to "cuda"
```

## Model specification

Before moving on to model inference chapter, let's know more about the converted model structure which is very important for model inference.

The converted model locates in the working directory like `mmdeploy_models/mmdet/onnx` in the previous example. It includes:

```
mmdeploy_models/mmdet/onnx
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- ***xxx*.json**: the necessary information for mmdeploy SDK

The whole package **mmdeploy_models/mmdet/onnx** is defined as **mmdeploy SDK model**, i.e., **mmdeploy SDK model** includes both backend model and inference meta information.

## Model inference

### Backend model inference

Take the previous converted `end2end.onnx` model as an example, you can use the following code to inference the model and visualize the results.

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = '../mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py'
model_cfg = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
device = 'cpu'
backend_model = ['mmdeploy_models/mmdet/onnx/end2end.onnx']
image = 'demo/demo.jpg'

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

img = cv2.imread('demo/demo.jpg')

# create a detector
detector = Detector(model_path='mmdeploy_models/mmdet/onnx',
                    device_name='cpu', device_id=0)
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

Besides python API, mmdeploy SDK also provides other FFI (Foreign Function Interface), such as C, C++, C#, Java and so on. You can learn their usage from [demos](https://github.com/open-mmlab/mmdeploy/tree/1.x/demo).

## Supported models

Please refer to [here](https://mmdeploy.readthedocs.io/en/1.x/04-supported-codebases/mmdet.html#supported-models) for the supported model list.
