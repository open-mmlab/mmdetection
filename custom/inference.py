import time
import json
from mmdet.apis import init_detector, inference_detector
from convert.utils import extract_bounding_boxes
import os
import argparse
# Specify the path to model config and checkpoint file
# config_file = 'configs/custom/my_retinanet_pvt-t_fpn_1x_coco.py'
# checkpoint_file = 'work_dirs/my_retinanet_pvt-t_fpn_1x_coco/epoch_7.pth'


# config_file = 'configs/custom_yolo/yolox_s_8x8_300e_coco.py'
# checkpoint_file = 'work_dirs/yolox_s_8x8_300e_coco/latest.pth'

# config_file = 'configs/custom/pvt_fst_v2/brummer.py'
# checkpoint_file = 'work_dirs/fst/latest.pth'


# inf_dir = 'data/fst/random_select_/random_select'
# inf_out_dir = 'data/fst/random_select_/fstv2_first'

# inf_dir = "/data/label_studio_datasets/forklift_person_id_84/images_test"
# inf_out_dir = '/data/label_studio_datasets/forklift_person_id_84/inference_result_annotations'

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
                 config_file,
                 checkpoint_file,
                 inf_dir,
                 inf_out_dir,
                threshold=50,
                output_file = None,
                 ):
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
        os.makedirs(inf_out_dir, exist_ok=True)
        self.inf_dir = inf_dir
        self.inf_out_dir = inf_out_dir
        self.output_file = output_file
        self.fps_logger = FPSLogger()
        self.run(threshold)



    
    def run(self, threshold = 50):
            results = []
            for imn in os.listdir(self.inf_dir):
                img = f'{self.inf_dir}/{imn}'
                self.fps_logger.start_record()
                result = inference_detector(self.model, img)
                results.append({img : extract_bounding_boxes(result, threshold)})
                self.fps_logger.end_record()
                # model.show_result(img, result)
                # self.model.show_result(img, result, out_file=f'{self.inf_out_dir}/{imn}', score_thr=0.5)
            if self.output_file:
                with open(self.output_file, 'w') as f:
                    json.dump(results, f)
            return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('--config_file', type=str, help='Path to model config file')
    parser.add_argument('--checkpoint_file', type=str, help='Path to checkpoint file')
    parser.add_argument('--inf_dir', type=str, help='Directory containing images for inference')
    parser.add_argument('--inf_out_dir', type=str, help='Directory to save inference results')
    parser.add_argument('--threshold', type=str, help='Directory to save inference results')
    parser.add_argument('--output_file', type=str, help='Directory to save inference results')

    args = parser.parse_args()
    if not args.threshold:
        args.threshold = 50
    Inference(
        config_file=args.config_file,
        checkpoint_file=args.checkpoint_file,
        inf_dir=args.inf_dir,
        inf_out_dir=args.inf_out_dir,
        threshold= args.threshold,
        output_file= args.output_file
    )
