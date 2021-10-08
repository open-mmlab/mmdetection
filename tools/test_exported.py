# Copyright (C) 2021 Intel Corporation
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

import argparse
import cv2
import mmcv
import numpy as np
import sys
import torch
from mmcv.parallel import DataContainer, collate

from mmdet.apis.inference import LoadImage
from mmdet.core import encode_mask_results
from mmdet.core.bbox.transforms import bbox2result
from mmdet.core.mask.transforms import mask2result
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.datasets.pipelines import Compose
from mmdet.utils import ExtendedDictAction


def postprocess(result, img_meta, num_classes=80, rescale=True):
    det_bboxes = result['boxes']
    det_labels = result['labels']
    det_masks = result.get('masks', None)
    det_texts = result.get('texts', None)

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
            img_size=(img_h, img_w))
        segm_results = encode_mask_results(segm_results)
        if det_texts is not None:
            return bbox_results, segm_results, det_texts
        else:
            return bbox_results, segm_results
    return bbox_results


def empty_result(num_classes=80, with_mask=False):
    bbox_results = [
        np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes)
    ]
    if with_mask:
        segm_results = [[] for _ in range(num_classes)]
        return bbox_results, segm_results
    return bbox_results


class VideoDataset:

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
    if args.update_config is not None:
        cfg.merge_from_dict(args.update_config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if backend == 'openvino':
        assert cfg.data.test.pipeline[1]['type'] == 'MultiScaleFlipAug'
        normalize_idx = [
            i for i, v in enumerate(cfg.data.test.pipeline[1]['transforms'])
            if v['type'] == 'Normalize'
        ][0]
        cfg.data.test.pipeline[1]['transforms'][normalize_idx]['mean'] = [
            0.0, 0.0, 0.0
        ]
        cfg.data.test.pipeline[1]['transforms'][normalize_idx]['std'] = [
            1.0, 1.0, 1.0
        ]
        cfg.data.test.pipeline[1]['transforms'][normalize_idx][
            'to_rgb'] = False
        print(cfg.data.test)

    if args.video is not None and args.show:
        dataset = VideoDataset(int(args.video), cfg.data)
        data_loader = iter(dataset)
        wait_key = 1
    else:
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        wait_key = -1

    # Valid classes + background.
    classes_num = len(dataset.CLASSES) + 1

    if backend == 'openvino':
        extra_args = {}
        if cfg.model.type == 'MaskTextSpotter':
            from mmdet.utils.deployment.openvino_backend import \
                MaskTextSpotterOpenVINO as Model
            extra_args['text_recognition_thr'] = cfg['model'].get(
                'roi_head', {}).get('text_thr', 0.0)
        else:
            from mmdet.utils.deployment.openvino_backend import \
                Detector as Model

        model = Model(
            args.model, cfg=cfg, classes=dataset.CLASSES, **extra_args)
    else:
        from mmdet.utils.deployment.onnxruntime_backend import ModelONNXRuntime
        model = ModelONNXRuntime(args.model, cfg=cfg, classes=dataset.CLASSES)

    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if torch.is_tensor(data['img'][0]):
            im_data = data['img'][0].cpu().numpy()
        elif isinstance(data['img'][0], DataContainer):
            im_data = data['img'][0].data[0].cpu().numpy()
        else:
            raise RuntimeError("Unknown image data type")

        try:
            result = model(im_data)
            result = postprocess(
                result,
                data['img_metas'][0].data[0],
                num_classes=classes_num,
                rescale=not args.show)
        except Exception as ex:
            print(f'\nException raised while processing item {i}:')
            print(ex)
            with_mask = hasattr(model.pt_model,
                                'with_mask') and model.pt_model.with_mask
            result = empty_result(num_classes=classes_num, with_mask=with_mask)
        results.append(result)

        if args.show:
            img_meta = data['img_metas'][0].data[0][0]

            norm_cfg = img_meta['img_norm_cfg']
            mean = np.array(norm_cfg['mean'], dtype=np.float32)
            std = np.array(norm_cfg['std'], dtype=np.float32)
            display_image = im_data[0].transpose(1, 2, 0)
            display_image = mmcv.imdenormalize(
                display_image, mean, std,
                to_bgr=norm_cfg['to_rgb']).astype(np.uint8)
            display_image = np.ascontiguousarray(display_image)

            h, w, _ = img_meta['img_shape']
            display_image = display_image[:h, :w, :]

            model.show(
                display_image,
                result,
                score_thr=args.score_thr,
                wait_time=wait_key)

        prog_bar.update()

    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(results, args.out)
    if args.eval:
        kwargs = cfg.get('evaluation', {})
        kwargs.pop('interval', None)
        kwargs.pop('gpu_collect', None)
        kwargs['metric'] = args.eval
        dataset.evaluate(results, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test model deployed to ONNX or OpenVINO')
    parser.add_argument('config', help='path to configuration file')
    parser.add_argument(
        'model',
        help='path to onnx model file or xml file in case of OpenVINO.')
    parser.add_argument(
        '--out', type=str, help='path to file with inference results')
    parser.add_argument(
        '--json_out',
        type=str,
        help='output result file name without extension')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal", "f1" for COCO, and "mAP", "recall" for PASCAL VOC'
    )
    parser.add_argument(
        '--video',
        default=None,
        help='run model on the video rather than the dataset')
    parser.add_argument(
        '--show', action='store_true', help='visualize results')
    parser.add_argument(
        '--score_thr',
        type=float,
        default=0.3,
        help='show only detections with confidence larger than the threshold')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=mmcv.DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--update_config',
        nargs='+',
        action=ExtendedDictAction,
        help='Update configuration file by parameters specified here.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
