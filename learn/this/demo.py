from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = '../../configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = '../../checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the reuslts
img = 'test.jpg'
result = inference_detector(model, img)
show_result(img, result, model.CLASSES, out_file='result.jpg')
