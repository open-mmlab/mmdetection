import argparse

import sys
import cv2
import mmcv
import numpy as np

from mmdet.core import coco_eval, results2json
from mmdet.core.bbox.transforms import bbox2result
from mmdet.core.mask.transforms import mask2result
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from mmcv.parallel import collate
from mmdet.datasets.pipelines import Compose
from mmdet.apis.inference import LoadImage


def postprocess(result, img_meta, num_classes=81, rescale=True):
    det_bboxes = result['boxes']
    det_labels = result['labels']
    det_masks = result.get('masks', None)

    if rescale:
        img_h, img_w = img_meta[0]['ori_shape'][:2]
        scale = img_meta[0]['scale_factor']
        det_bboxes[:, :4] /= scale
    else:
        img_h, img_w = img_meta[0]['img_shape'][:2]

    det_bboxes[:, 0:4:2] = np.clip(det_bboxes[:, 0:4:2], 0, img_w - 1)
    det_bboxes[:, 1:4:2] = np.clip(det_bboxes[:, 1:4:2], 0, img_h - 1)

    bbox_results = bbox2result(det_bboxes, det_labels, num_classes)
    if det_masks is not None:
        segm_results = mask2result(
            det_bboxes,
            det_labels,
            det_masks,
            num_classes,
            mask_thr_binary=0.5,
            rle=True,
            full_size=True,
            img_size=(img_h, img_w))
        return bbox_results, segm_results
    return bbox_results


def empty_result(num_classes=81, with_mask=False):
    bbox_results = [
        np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes - 1)
    ]
    if with_mask:
        segm_results = [[] for _ in range(num_classes - 1)]
        return bbox_results, segm_results
    return bbox_results


class VideoDataset(object):
    def __init__(self, path, cfg, device='cpu'):
        self.path = path
        self.video = cv2.VideoCapture(self.path)
        assert self.video.isOpened()
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cfg = cfg
        self.device = device

        # build the data pipeline
        self.test_pipeline = [LoadImage()] + self.cfg.test.pipeline[1:]
        self.test_pipeline = Compose(self.test_pipeline)

    def __getitem__(self, idx):
        status, img = self.video.read()
        if not status:
            self.video.release()
            raise StopIteration

        data = dict(img=img)
        data = self.test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        return data

    def __len__(self):
        return sys.maxsize


def main(args):
    if args.model.endswith('.onnx'):
        backend = 'onnx'
    elif args.model.endswith('.xml'):
        backend = 'openvino'
    else:
        raise ValueError('Unknown model type.')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if backend == 'openvino':
        assert cfg.data.test.pipeline[1]['type'] == 'MultiScaleFlipAug'
        normalize_idx = [i for i, v in enumerate(cfg.data.test.pipeline[1]['transforms']) if v['type'] == 'Normalize'][0]
        cfg.data.test.pipeline[1]['transforms'][normalize_idx]['mean'] = [0.0, 0.0, 0.0]
        cfg.data.test.pipeline[1]['transforms'][normalize_idx]['std'] = [1.0, 1.0, 1.0]
        cfg.data.test.pipeline[1]['transforms'][normalize_idx]['to_rgb'] = False
        print(cfg.data.test)

    if args.video is not None and args.show:
        dataset = VideoDataset(int(args.video), cfg.data)
        data_loader = iter(dataset)
        wait_key = 1
    else:
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        wait_key = -1

    # Valid classes + background.
    classes_num = len(dataset.CLASSES) + 1

    if backend == 'openvino':
        from mmdet.utils.openvino import DetectorOpenVINO
        model = DetectorOpenVINO(args.model,
                                 args.model[:-3] + 'bin',
                                 mapping_file_path=args.model[:-3] + 'mapping',
                                 with_detection_output=args.with_detection_output,
                                 cfg=cfg,
                                 classes=dataset.CLASSES)
    else:
        from mmdet.utils.onnxruntime_backend import ONNXModel
        model = ONNXModel(args.model, cfg=cfg, classes=dataset.CLASSES)

    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        im_data = data['img'][0].cpu().numpy()
        try:
            result = model(im_data)
            result = postprocess(
                result,
                data['img_meta'][0].data[0],
                num_classes=classes_num,
                rescale=not args.show)
        except Exception as ex:
            print('\nException raised while processing item {}:'.format(i))
            print(ex)
            result = empty_result(
                num_classes=classes_num,
                with_mask=model.pt_model.with_mask)
        results.append(result)

        if args.show:
            model.show(data, result, score_thr=args.score_thr, wait_time=wait_key)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    print('')
    print('Writing results to {}'.format(args.out))
    mmcv.dump(results, args.out)

    eval_types = args.eval
    if eval_types:
        print('Starting evaluate {}'.format(' and '.join(eval_types)))
        if eval_types == ['proposal_fast']:
            result_file = args.out
            coco_eval(result_file, eval_types, dataset.coco)
        else:
            if not isinstance(results[0], dict):
                result_files = results2json(dataset, results, args.out)
                coco_eval(result_files, eval_types, dataset.coco)
            else:
                for name in results[0]:
                    print('\nEvaluating {}'.format(name))
                    outputs_ = [out[name] for out in results]
                    result_file = args.out + '.{}'.format(name)
                    result_files = results2json(dataset, outputs_,
                                                result_file)
                    coco_eval(result_files, eval_types, dataset.coco)

    # Save predictions in the COCO json format
    if args.json_out:
        if not isinstance(results[0], dict):
            results2json(dataset, results, args.json_out)
        else:
            for name in results[0]:
                outputs_ = [out[name] for out in results]
                result_file = args.json_out + '.{}'.format(name)
                results2json(dataset, outputs_, result_file)


def parse_args():
    parser = argparse.ArgumentParser(description='Test model deployed to ONNX or OpenVINO')
    parser.add_argument('config', help='path to configuration file')
    parser.add_argument('model', help='path to onnx model file or xml file in case of OpenVINO.')
    parser.add_argument('--out', type=str, help='path to file with inference results')
    parser.add_argument('--json_out', type=str, help='output result file name without extension')
    parser.add_argument('--eval', type=str, nargs='+',
                        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
                        help='eval types')
    parser.add_argument('--video', default=None, help='run model on the video rather than the dataset')
    parser.add_argument('--show', action='store_true', help='visualize results')
    parser.add_argument('--score_thr', type=float, default=0.3,
                        help='show only detections with confidence larger than the threshold')
    openvino_args = parser.add_argument_group('OpenVINO-related arguments')
    openvino_args.add_argument('--with_detection_output', action='store_true',
                               help='expect DetectionOutput operation at the end of the OpenVINO net')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
