import asyncio
import os
import re
from argparse import ArgumentParser

from numpy import array, float32

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('model_name', help='serve model name')
    parser.add_argument(
        '--inference_adr',
        default='127.0.0.1:8080',
        help='serve inference port address')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        score_thr=args.score_thr,
        title='pytorch_result')


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


def res_format(str_res, model):
    res_str = re.sub('[\\s\\[\\]\\{\\}\\"]*', '', str_res)
    anchor_set = re.split(',|:', res_str)
    cls_set = [[] for i in range(model.CLASSES.__len__())]
    for i in range(0, anchor_set.__len__(), 7):
        cls_set[model.CLASSES.index(anchor_set[i])].append([
            float(anchor_set[i + 1]),
            float(anchor_set[i + 2]),
            float(anchor_set[i + 3]),
            float(anchor_set[i + 4]),
            float(anchor_set[i + 6])
        ])
    result = []
    for cls in cls_set:
        if cls.__len__() == 0:
            result.append(array(cls, dtype=float32).reshape((0, 5)))
        else:
            result.append(array(cls, dtype=float32))
    return result


def serve_inference(args):
    tmp_res = os.popen('curl http://' + args.inference_adr + '/predictions/' +
                       args.model_name + ' -T ' + args.img).read(-1)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    format_result = res_format(tmp_res, model)
    show_result_pyplot(
        model,
        args.img,
        format_result,
        score_thr=args.score_thr,
        title='serve_result')


def compare_res(args):
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
    serve_inference(args)


if __name__ == '__main__':
    args = parse_args()
    compare_res(args)
