from __future__ import division
import argparse
import os
import mmcv

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.utils.utils import get_files


def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('config_file', help='test config file path')
    parser.add_argument(
        '--checkpoint_file', help='the checkpoint file to resume from')
    parser.add_argument(
        '--img_folder', help='test image folder')
    parser.add_argument(
        '--output_folder', help='output image folder')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    model = init_detector(args.config_file, args.checkpoint_file, device='cuda:0')
    test_imgs = get_files(args.img_folder, ".tif")
    print(len(test_imgs))

    for img in test_imgs:
        result = inference_detector(model, os.path.join(args.img_folder, img))
        show_result_pyplot(os.path.join(args.img_folder, img), result, model.CLASSES,
                           bbox_color='red',
                           text_color='red',
                           thickness=3,
                           score_thr=0.4,
                           out_file=os.path.join(args.output_folder, img))


if __name__ == '__main__':
    main()