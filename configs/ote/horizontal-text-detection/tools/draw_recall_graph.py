# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

""" This script draws a graph of the detected words number (recall)
    depending on their width. It helps to see the detection quality of the
    small, normal or large inscriptions. Also for your convenience you may
    visualize the detections straight away."""

import argparse
from os.path import exists
import subprocess

import mmcv

from mmdet.datasets import build_dataset # pylint: disable=import-error
from mmdet.core.evaluation.text_evaluation import text_eval # pylint: disable=import-error


def parse_args():
    """ Parses input arguments. """
    parser = argparse.ArgumentParser(
        description='This script draws a histogram of the detected words '
                    'number (recall) depending on their width. It helps to '
                    'see the detection quality of the small, normal or large '
                    'inscriptions. Also for your convenience you may '
                    'visualize the detections straight away.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('snapshot', help='path to snapshot to be tested')
    parser.add_argument('--draw_graph', action='store_true', help='draw histogram of recall')
    parser.add_argument('--visualize', action='store_true', help='show detection result on images')
    args = parser.parse_args()
    return args


def main():
    """ Main function. """
    args = parse_args()

    detection_file = 'horizontal_text_detection'
    if not exists(f'{detection_file}.bbox.json'):
        subprocess.run(
            f'python ../../../../../external/mmdetection/tools/test.py'
            f' {args.config} {args.snapshot}'
            f' --options jsonfile_prefix={detection_file}'
            f' --format-only',
            check=True, shell=True
        )

    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)

    results = mmcv.load(f'{detection_file}.bbox.json')
    coco = dataset.coco
    coco_dets = coco.loadRes(results)
    predictions = coco_dets.imgToAnns
    gt_annotations = coco.imgToAnns

    if args.visualize:
        img_paths = [dataset.img_prefix + image['file_name']
                     for image in coco_dets.dataset['images']]
    else:
        img_paths = None

    recall, precision, hmean, _ = text_eval(
        predictions, gt_annotations,
        cfg.test_cfg.score_thr,
        images=img_paths,
        show_recall_graph=args.draw_graph)
    print('Text detection recall={:.4f} precision={:.4f} hmean={:.4f}'.
          format(recall, precision, hmean))


if __name__ == '__main__':
    main()
