from mmdet.apis import init_detector, inference_detector
import mmcv
import os

# Specify the path to model config and checkpoint file
config_file = 'configs/custom/my_retinanet_pvt-t_fpn_1x_coco.py'
checkpoint_file = 'work_dirs/my_retinanet_pvt-t_fpn_1x_coco/epoch_7.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
inf_dir = "data/val/image_2"
#imn = "img_2022_03_31_09_45_42t163637.jpg"
for imn in os.listdir(inf_dir):
    img = f'{inf_dir}/{imn}' # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)
    # visualize the results in a new window
    model.show_result(img, result)
    # or save the visualization results to image files
    model.show_result(img, result, out_file=f'data/val/inference/{imn}', score_thr=0.5)

# # test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)