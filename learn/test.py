from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

print("hello")

config_file = '/home/507/myfiles/code/mmdetection/myfile/from_zero/config/fasterrcnn/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = '/home/507/myfiles/code/mmdetection/myfile/from_zero/result/fasterrcnn/try-1/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)

# or save the visualization results to image files
show_result(img, result, model.CLASSES, out_file='result2.jpg')

