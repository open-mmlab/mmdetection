import time

from mmdet.apis import init_detector, inference_detector
import mmcv
import os

# Specify the path to model config and checkpoint file
# config_file = 'configs/custom/my_retinanet_pvt-t_fpn_1x_coco.py'
# checkpoint_file = 'work_dirs/my_retinanet_pvt-t_fpn_1x_coco/epoch_7.pth'

config_file = 'configs/custom/pvt_brummer_v2/brummer.py'
checkpoint_file = 'work_dirs/brummer/latest.pth'

# config_file = 'configs/custom_yolo/yolox_s_8x8_300e_coco.py'
# checkpoint_file = 'work_dirs/yolox_s_8x8_300e_coco/latest.pth'

# config_file = 'configs/custom/pvt_fst_v2/brummer.py'
# checkpoint_file = 'work_dirs/fst/latest.pth'


# inf_dir = 'data/fst/random_select_/random_select'
# inf_out_dir = 'data/fst/random_select_/fstv2_first'

inf_dir = "data/brummer/val/image_2"
inf_out_dir = 'data/brummer/val/brummer_v2_debug'

class FPSLogger():
    def __init__(self):
        self.tottime = 0.
        self.count = 0
        self.last_record = 0.
        self.last_print = time.time()
        self.interval = 10

    def start_record(self):
        self.last_record = time.time()

    def end_record(self):
        self.tottime += time.time() - self.last_record
        self.count += 1
        self.print_fps()


    def print_fps(self):
        if time.time() - self.last_print > self.interval:
            print(f"Inference running at {self.count / self.tottime:.3f} FPS")
            self.last_print = time.time()

class Inference():
    def __init__(self,
                 config_file=config_file,
                 checkpoint_file=checkpoint_file,
                 inf_dir=inf_dir,
                 inf_out_dir=inf_out_dir):
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
        os.makedirs(inf_out_dir, exist_ok=True)
        self.inf_dir = inf_dir
        self.inf_out_dir = inf_out_dir
        self.fps_logger = FPSLogger()
        self.run()


    def run(self):
        for imn in os.listdir(self.inf_dir):
            img = f'{inf_dir}/{imn}'
            self.fps_logger.start_record()
            result = inference_detector(self.model, img)
            self.fps_logger.end_record()
            # model.show_result(img, result)
            self.model.show_result(img, result, out_file=f'{self.inf_out_dir}/{imn}', score_thr=0.5)



if __name__ == "__main__":
    Inference()
