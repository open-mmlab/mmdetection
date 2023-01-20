# Copyright (c) OpenMMLab. All rights reserved.
import os
import urllib
from argparse import ArgumentParser

import mmcv
import torch
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, scandir

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def get_file_list(source_root: str) -> [list, dict]:
    """Get file list.

    Args:
        source_root (str): image or video source path

    Return:
        source_file_path_list (list): A list for all source file.
        source_type (dict): Source type: file or url or dir.
    """
    is_dir = os.path.isdir(source_root)
    is_url = source_root.startswith(('http:/', 'https:/'))
    is_file = os.path.splitext(source_root)[-1].lower() in IMG_EXTENSIONS

    source_file_path_list = []
    if is_dir:
        # when input source is dir
        for file in scandir(source_root, IMG_EXTENSIONS, recursive=True):
            source_file_path_list.append(os.path.join(source_root, file))
    elif is_url:
        # when input source is url
        filename = os.path.basename(
            urllib.parse.unquote(source_root).split('?')[0])
        file_save_path = os.path.join(os.getcwd(), filename)
        print(f'Downloading source file to {file_save_path}')
        torch.hub.download_url_to_file(source_root, file_save_path)
        source_file_path_list = [file_save_path]
    elif is_file:
        # when input source is single image
        source_file_path_list = [source_root]
    else:
        print('Cannot find image file.')

    source_type = dict(is_dir=is_dir, is_url=is_url, is_file=is_file)

    return source_file_path_list, source_type


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--dataset', type=str, help='dataset name to load the text embedding')
    parser.add_argument(
        '--class-name', nargs='+', type=str, help='custom class names')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    if not os.path.exists(args.out_dir) and not args.show:
        os.mkdir(args.out_dir)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # get file list
    files, source_type = get_file_list(args.img)
    from detic.utils import (get_class_names, get_text_embeddings,
                             reset_cls_layer_weight)

    # class name embeddings
    if args.class_name:
        dataset_classes = args.class_name
    elif args.dataset:
        dataset_classes = get_class_names(args.dataset)
    embedding = get_text_embeddings(
        dataset=args.dataset, custom_vocabulary=args.class_name)
    visualizer.dataset_meta['classes'] = dataset_classes
    reset_cls_layer_weight(model, embedding)

    # start detector inference
    progress_bar = ProgressBar(len(files))
    for file in files:
        result = inference_detector(model, file)

        img = mmcv.imread(file)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        progress_bar.update()

        visualizer.add_datasample(
            filename,
            img,
            data_sample=result,
            draw_gt=False,
            show=args.show,
            wait_time=0,
            out_file=out_file,
            pred_score_thr=args.score_thr)

    if not args.show:
        print_log(
            f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
