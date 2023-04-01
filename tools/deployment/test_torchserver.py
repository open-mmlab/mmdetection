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
        help='Address and port of the inference server')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def parse_result(input, model_class):
    bbox = []
    label = []
    score = []
    for anchor in input:
        bbox.append(anchor['bbox'])
        label.append(model_class.index(anchor['class_name']))
        score.append([anchor['score']])
    bboxes = np.append(bbox, score, axis=1)
    labels = np.array(label)
    result = bbox2result(bboxes, labels, len(model_class))
    return result


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    model_result = inference_detector(model, args.img)
    for i, anchor_set in enumerate(model_result):
        anchor_set = anchor_set[anchor_set[:, 4] >= 0.5]
        model_result[i] = anchor_set
    # show the results
    show_result_pyplot(
        model,
        args.img,
        model_result,
        score_thr=args.score_thr,
        title='pytorch_result')
    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    with open(args.img, 'rb') as image:
        response = requests.post(url, image)
    server_result = parse_result(response.json(), model.CLASSES)
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
