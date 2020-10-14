# Verification

To verify whether MMDetection and the required environment are installed correctly, we can run sample python codes to initialize a detector and inference a demo image:

From the command line, type:

```bash
python
```

then enter the following code:

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, device=device)
# inference the demo image
inference_detector(model, 'demo/demo.jpg')
```

The above code is supposed to run successfully upon you finish the installation.
