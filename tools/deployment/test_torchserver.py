from argparse import ArgumentParser

import numpy as np
import requests

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.core import bbox2result


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


def parse_result(tmp_res, model):
    bbox = []
    label = []
    score = []
    for anchor in tmp_res:
        bbox.append(anchor['bbox'])
        label.append(model.CLASSES.index(anchor['class_name']))
        score.append([anchor['score']])
    bboxes = np.append(bbox, score, axis=1)
    labels = np.array(label)
    result = bbox2result(bboxes, labels, len(model.CLASSES))
    return result


def result_filter(model_result):
    filted_result = []
    for anchor_set in model_result:
        delete_list = []
        for i in range(anchor_set.shape[0]):
            if anchor_set[i][4] < 0.5:
                delete_list.append(i)
        anchor_set = np.delete(anchor_set, delete_list, axis=0)
        filted_result.append(anchor_set)
    return filted_result


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    model_result = inference_detector(model, args.img)
    model_result = result_filter(model_result)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        model_result,
        score_thr=args.score_thr,
        title='pytorch_result')
    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    tmp_res = requests.post(url, open(args.img, 'rb'))
    server_result = parse_result(tmp_res.json(), model)
    show_result_pyplot(
        model,
        args.img,
        server_result,
        score_thr=args.score_thr,
        title='server_result')

    for i in range(len(model.CLASSES)):
        assert np.allclose(model_result[i], server_result[i])


if __name__ == '__main__':
    args = parse_args()
    main(args)
