# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet.apis import DetInferencer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        'model',
        type=str,
        help='Config or checkpoint .pth file or the model name defined '
        'in metafile. The model configuration file will try to read '
        'from .pth if the parameter is a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--img-out-dir',
        type=str,
        default='outputs',
        help='Output directory of images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-image',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--pred-out-file',
        type=str,
        default='',
        help='File to save the inference results. '
        'Currently only supports json suffix.')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')

    call_args = vars(parser.parse_args())

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, which is '
                  'automatically replaced by the --weights parameter')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    no_save_image = call_args.pop('no_save_image')
    if no_save_image and not call_args['show'] and call_args[
            'pred_out_file'] == '':
        warnings.warn(
            'It doesn\'t make sense to neither save the prediction '
            'result nor display it. Force set args.no-save-image to False')
        no_save_image = False
    if no_save_image:
        call_args['img_out_dir'] = ''

    if call_args['pred_out_file'] != '':
        assert call_args['pred_out_file'].endswith('.json'), \
            f'The --pred-out-file: {call_args["pred_out_file"]} ' \
            'must be a json file.'

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def main():
    init_args, call_args = parse_args()
    inferencer = DetInferencer(**init_args)
    inferencer(**call_args)

    if call_args['img_out_dir'] != '':
        print_log('\nVisualized results have been saved at '
                  f'{call_args["img_out_dir"]}')
    if call_args['pred_out_file'] != '':
        print_log('Predicted Results have been saved at '
                  f'{call_args["pred_out_file"]}')


if __name__ == '__main__':
    main()
