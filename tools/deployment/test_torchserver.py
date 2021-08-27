import asyncio
from argparse import ArgumentParser

import numpy as np
import requests

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('model_name', help='The model name in the server')
    parser.add_argument(
        '--inference-addr',
        default='127.0.0.1:8080',
        help='server inference port address')
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


def main(args, visual_result):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    if visual_result:
        show_result_pyplot(
            model,
            args.img,
            result,
            score_thr=args.score_thr,
            title='pytorch_result')
    return result


async def async_main(args, visual_result):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    if visual_result:
        show_result_pyplot(
            model, args.img, result[0], score_thr=args.score_thr)
    return result


def parse_result(tmp_res, model):
    cls_set = [[] for i in range(len(model.CLASSES))]
    for anchor in tmp_res:
        cls_set[model.CLASSES.index(list(anchor.keys())[0])]\
            .append([*list(anchor.values())[0], anchor['score']])
    result = []
    for cls in cls_set:
        if len(cls) == 0:
            result.append(np.array(cls, dtype=np.float32).reshape((0, 5)))
        else:
            result.append(np.array(cls, dtype=np.float32))
    return result


def serve_inference(args, visual_result):
    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    tmp_res = requests.post(url, open(args.img, 'rb'))
    model = init_detector(args.config, args.checkpoint, device=args.device)
    server_result = parse_result(tmp_res.json(), model)
    if visual_result:
        show_result_pyplot(
            model,
            args.img,
            server_result,
            score_thr=args.score_thr,
            title='server_result')
    return server_result


def compare_res(args, visual_result=True):
    if args.async_test:
        torch_res = asyncio.run(async_main(args), visual_result)
    else:
        torch_res = main(args, visual_result)
    server_result = serve_inference(args, visual_result)
    assert torch_res == server_result


if __name__ == '__main__':
    args = parse_args()
    compare_res(args)
