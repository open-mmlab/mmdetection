import argparse
import os.path as osp

import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet.core.evaluation import eval_map
from mmdet.core.visualization.image import imshow_det_bboxes
from mmdet.datasets import build_dataset


def visualize(img,
              annotation,
              result,
              class_names=None,
              show=True,
              wait_time=0,
              out_file=None):
    imshow_det_bboxes(
        img,
        annotation['bboxes'],
        annotation['labels'],
        class_names=class_names,
        bbox_color=(255, 102, 61),
        text_color=(255, 102, 61),
        show=False)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms=segm_result,
        class_names=class_names,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
        show=show,
        wait_time=wait_time,
        out_file=out_file)


class ResultVisualizer(object):

    def __init__(self,
                 dataset,
                 results,
                 topk=20,
                 show=False,
                 wait_time=0,
                 show_dir='work_dir'):
        self.topk = topk
        self.dataset = dataset
        assert self.topk > 0
        if (self.topk * 2) > len(self.dataset):
            self.topk = len(dataset) // 2
        self.results = results
        self.CLASSES = self.dataset.CLASSES
        self.show = show
        self.wait_time = wait_time
        self.show_dir = show_dir
        self.annotations = [
            self.dataset.get_ann_info(i) for i in range(len(self.dataset))
        ]

    def _eval_fn(self, det_result, annotation):
        iou_thrs = np.linspace(
            .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        mean_aps = []
        for thr in iou_thrs:
            mean_ap, _ = eval_map([det_result], [annotation], iou_thr=thr)
            mean_aps.append(mean_ap)
        return sum(mean_aps) / len(mean_aps)

    def _save_show_images(self, mAPs, out_dir):
        mmcv.mkdir_or_exist(out_dir)
        for mAP in mAPs:
            self._show_image_gts_results(mAP[0], mAP[1], out_dir)

    def _show_image_gts_results(self, index, value, out_dir=None):
        img_info = self.dataset[index]
        annotation = self.annotations[index]
        result = self.results[index]
        img = img_info['img']
        filename = img_info['img_info']['filename']
        if img_info['img_prefix'] is not None:
            filename = osp.join(img_info['img_prefix'], filename)
        else:
            filename = img_info['filename']
        fname, name = osp.splitext(osp.basename(filename))
        save_filename = fname + '_' + str(round(value, 3)) + name
        out_file = osp.join(out_dir, save_filename)
        visualize(img, annotation, result, self.CLASSES, self.show,
                  self.wait_time, out_file)

    def evaluate(self):
        _mAPs = {}
        for i, (result,
                annotation) in enumerate(zip(self.results, self.annotations)):
            mAP = self._eval_fn(result, annotation)
            _mAPs[i] = mAP

        # descending select topk image
        _mAPs = list(sorted(_mAPs.items(), key=lambda kv: kv[1]))

        good_mAPs = _mAPs[-self.topk:]
        bad_mAPs = _mAPs[:self.topk]

        good_dir = osp.abspath(osp.join(self.show_dir, 'good'))
        bad_dir = osp.abspath(osp.join(self.show_dir, 'bad'))
        self._save_show_images(good_mAPs, good_dir)
        self._save_show_images(bad_mAPs, bad_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test pkl result')
    parser.add_argument(
        'show_dir', help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=0,
        help='the interval of show (s), 0 is block')
    parser.add_argument(
        'eval',
        type=str,
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--topk',
        default=20,
        type=int,
        help='Saved Number of the highest topk '
        'and lowest topk after index sorting')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mmcv.check_file_exist(args.prediction_path)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    cfg.data.test.pop('samples_per_gpu', 0)
    cfg.data.test.pipeline = [cfg.data.test.pipeline[0]]
    assert cfg.data.test.pipeline[0]['type'] in [
        'LoadImageFromFile', 'LoadMultiChannelImageFromFiles'
    ]
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.prediction_path)
    result_visualizer = ResultVisualizer(
        dataset,
        outputs,
        topk=args.topk,
        show=args.show,
        wait_time=args.wait_time,
        show_dir=args.show_dir)
    result_visualizer.evaluate()


if __name__ == '__main__':
    main()
