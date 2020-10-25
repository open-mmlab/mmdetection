from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import matplotlib.pyplot as plt

config_file = '../configs/faster_rcnn_r50_fpn_1x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = 'demo.jpg'
result = inference_detector(model, img)

# show the results
show_result_pyplot(img, result, model.CLASSES)