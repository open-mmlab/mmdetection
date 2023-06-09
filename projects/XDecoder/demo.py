# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmengine.config import Config
from mmengine.logging import print_log

from mmdet.apis import DetInferencer
from projects.XDecoder.xdecoder.inference import (
    ImageCaptionInferencer, RefImageCaptionInferencer,
    TextToImageRegionRetrievalInferencer)

TASKINFOS = {
    'semseg': DetInferencer,
    'ref-seg': DetInferencer,
    'instance': DetInferencer,
    'panoptic': DetInferencer,
    'caption': ImageCaptionInferencer,
    'ref-caption': RefImageCaptionInferencer,
    'retrieval': TextToImageRegionRetrievalInferencer,
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument('model', type=str, help='Config file name')
    parser.add_argument('--weights', help='Checkpoint file')
    parser.add_argument('--texts', help='text prompt')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['ade20k', 'coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')

    # only for instance segmentation
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.5,
        help='bbox score threshold')
    # only for panoptic segmentation
    parser.add_argument(
        '--stuff-texts',
        help='text prompt for stuff name in panoptic segmentation')

    call_args = vars(parser.parse_args())
    if call_args['no_save_vis']:
        call_args['out_dir'] = ''

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def main():
    init_args, call_args = parse_args()

    cfg = Config.fromfile(init_args['model'])
    task = cfg.model.head.task
    assert task in TASKINFOS

    inferencer = TASKINFOS[task](**init_args)

    if task != 'caption':
        assert call_args[
            'texts'] is not None, f'text prompts is required for {task}'
        if task != 'panoptic':
            call_args.pop('stuff_texts')
    else:
        call_args.pop('texts')
        call_args.pop('stuff_texts')

    inferencer(**call_args)

    if call_args['out_dir'] != '' and not call_args['no_save_vis']:
        print_log(f'results have been saved at {call_args["out_dir"]}')


if __name__ == '__main__':
    main()
