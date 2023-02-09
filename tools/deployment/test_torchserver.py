import os
from argparse import ArgumentParser

import mmcv
import requests
import torch
from mmengine.structures import InstanceData

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample


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
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='output directory to save drawn results.')
    args = parser.parse_args()
    return args


def align_ts_output(inputs, metainfo, device):
    bboxes = []
    labels = []
    scores = []
    for i, pred in enumerate(inputs):
        bboxes.append(pred['bbox'])
        labels.append(pred['class_label'])
        scores.append(pred['score'])
    pred_instances = InstanceData(metainfo=metainfo)
    pred_instances.bboxes = torch.tensor(
        bboxes, dtype=torch.float32, device=device)
    pred_instances.labels = torch.tensor(
        labels, dtype=torch.int64, device=device)
    pred_instances.scores = torch.tensor(
        scores, dtype=torch.float32, device=device)
    ts_data_sample = DetDataSample(pred_instances=pred_instances)
    return ts_data_sample


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    pytorch_results = inference_detector(model, args.img)
    keep = pytorch_results.pred_instances.scores >= args.score_thr
    pytorch_results.pred_instances = pytorch_results.pred_instances[keep]

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    # show the results
    img = mmcv.imread(args.img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    pt_out_file = None
    ts_out_file = None
    if args.work_dir is not None:
        os.makedirs(args.work_dir, exist_ok=True)
        pt_out_file = os.path.join(args.work_dir, 'pytorch_result.png')
        ts_out_file = os.path.join(args.work_dir, 'torchserve_result.png')
    visualizer.add_datasample(
        'pytorch_result',
        img.copy(),
        data_sample=pytorch_results,
        draw_gt=False,
        out_file=pt_out_file,
        show=True,
        wait_time=0)

    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    with open(args.img, 'rb') as image:
        response = requests.post(url, image)
    metainfo = pytorch_results.pred_instances.metainfo
    ts_results = align_ts_output(response.json(), metainfo, args.device)

    visualizer.add_datasample(
        'torchserve_result',
        img,
        data_sample=ts_results,
        draw_gt=False,
        out_file=ts_out_file,
        show=True,
        wait_time=0)

    assert torch.allclose(pytorch_results.pred_instances.bboxes,
                          ts_results.pred_instances.bboxes)
    assert torch.allclose(pytorch_results.pred_instances.labels,
                          ts_results.pred_instances.labels)
    assert torch.allclose(pytorch_results.pred_instances.scores,
                          ts_results.pred_instances.scores)


if __name__ == '__main__':
    args = parse_args()
    main(args)
