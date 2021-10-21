import csv
import os.path as osp
import warnings
from collections import defaultdict

import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


def list_from_csv(filename):
    item_list = defaultdict(list)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        i = -1
        for line in reader:
            i += 1
            if i == 0:
                continue
            # else:
            elif i < 10:  # for debug
                img_id = line[0]
                label = line[2]
                x_min = line[4]
                x_max = line[5]
                y_mix = line[6]
                y_max = line[7]
                IsOccluded = True if line[8] == 1 else False
                IsTruncated = True if line[9] == 1 else False
                IsGroupOf = True if line[10] == 1 else False
                IsDepiction = True if line[11] == 1 else False
                IsInside = True if line[12] == 1 else False
                # ann[img_id].append()
                item_list[img_id].append(
                    dict(
                        bbox=[x_min, x_max, y_mix, y_max],
                        label=label,
                        IsOccluded=IsOccluded,
                        IsTruncated=IsTruncated,
                        IsGroupOf=IsGroupOf,
                        IsDepiction=IsDepiction,
                        IsInside=IsInside))
    return item_list


@DATASETS.register_module()
class OpenImagesDataset(CustomDataset):

    def __init__(self, label_csv_path='', min_size=None, **kwargs):
        super(OpenImagesDataset, self).__init__(**kwargs)
        self.min_size = min_size
        self.cat2label = {}
        self.CLASSES = self.get_classes_from_csv(label_csv_path)
        self.ann = self.get_ann_info(1)

    def get_classes_from_csv(self, label_csv_path):
        """class-descriptions-boxable.csv."""
        self.label_dict = defaultdict(str)
        label_list = []

        with open(label_csv_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                self.label_dict[line[0]] = line[1]
                label_list.append(line[1])
                self.cat2label[line[1]] = int(line[2])
        return label_list

    def load_annotations(self, ann_file):
        """Load annotation from csv style ann_file.

        Args:
            ann_file (str): .

        Returns:
            dict[list[dict]]: Annotation info from csv file.
        """

        data_infos = []
        self.ann_infos = list_from_csv(ann_file)
        img_ids = list(self.ann_infos.keys())
        for img_id in img_ids:
            filename = f'image/{img_id}.jpg'
            data_infos.append(dict(id=img_id, filename=filename))

        return data_infos

    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]['id']
        width = self.data_infos[idx]['width']
        height = self.data_infos[idx]['height']
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        IsOccludeds = []
        IsTruncateds = []
        IsGroupOfs = []
        IsDepictions = []
        IsInsides = []
        for obj in self.ann_infos[img_id]:
            name = self.label_dict[obj['label']]
            # label begin from 1
            label = self.cat2label[name] - 1
            bbox = [
                int(float(obj['bbox'][0]) * width),
                int(float(obj['bbox'][1]) * width),
                int(float(obj['bbox'][2]) * height),
                int(float(obj['bbox'][3]) * height)
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

            # Other parameters
            IsOccludeds.append(obj['IsOccluded'])
            IsTruncateds.append(obj['IsTruncated'])
            IsGroupOfs.append(obj['IsGroupOf'])
            IsDepictions.append(obj['IsDepiction'])
            IsInsides.append(obj['IsInside'])
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            # TODO: Understand why uses
            # `np.array(bboxes, ndmin=2) - 1`
            # in VOC(xml) dataset?
            bboxes = np.array(bboxes)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore)
            labels_ignore = np.array(labels_ignore)
            print()
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        from collections import OrderedDict
        from mmcv.utils import print_log

        from mmdet.core import eval_map

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)

        return eval_results


#
@DATASETS.register_module()
class OpenImagesChallengeDataset(CustomDataset):
    """需要增加一个 get father 这么一个参数."""

    def __init__(self, label_csv_path='', min_size=None, **kwargs):
        super(OpenImagesChallengeDataset, self).__init__(**kwargs)
        self.min_size = min_size
        self.CLASSES = self.get_classes_from_csv(label_csv_path)
        # for debug
        # self.ann = self.get_ann_info(1)

    def get_classes_from_csv(self, label_csv_path):
        """class-descriptions-boxable.csv."""
        self.label_dict = defaultdict(str)
        label_list = []
        self.cat2label = {}
        if self.data_root is not None:
            label_csv_path = osp.join(self.data_root, label_csv_path)
        with open(label_csv_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                self.label_dict[line[0]] = line[1]
                label_list.append(line[1])
                self.cat2label[line[1]] = int(line[2])
        return label_list

    def load_annotations(self, ann_file):
        print('load annotation begin', flush=True)
        with open(ann_file) as f:
            lines = f.readlines()
        i = 0
        ann_infos = []
        while i < len(lines):
            bboxes = []
            labels = []
            is_group_ofs = []
            filename = lines[i].rstrip()
            i += 2
            img_gt_size = int(lines[i])
            i += 1
            for j in range(img_gt_size):
                sp = lines[i + j].split()
                bboxes.append(
                    [float(sp[1]),
                     float(sp[2]),
                     float(sp[3]),
                     float(sp[4])])
                labels.append(int(sp[0]))
                is_group_ofs.append(True if sp[5] == 1 else False)
            i += img_gt_size

            gt_bboxes = np.array(bboxes, dtype=np.float32)
            # need -1?
            gt_labels = np.array(labels, dtype=np.int64)
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

            img_info = dict(filename=filename)
            ann_info = dict(
                bboxes=gt_bboxes,
                labels=gt_labels,
                bboxes_ignore=gt_bboxes_ignore,
            )
            ann_infos.append(dict(img_info=img_info, ann_info=ann_info))

        print('load annotation end', flush=True)
        return ann_infos

    def prepare_train_img(self, idx):
        ann_info = self.data_infos[idx]
        results = dict(
            img_info=ann_info['img_info'],
            ann_info=ann_info['ann_info'],
        )
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        ann_info = self.data_infos[idx]
        results = dict(img_info=ann_info['img_info'])
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn('OpenImageDatasets does not support '
                          'filtering empty gt images.')
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            # Due to openimage dataset do not have image information in
            # annotations
            # if min(img_info['width'], img_info['height']) >= min_size:
            valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            # img_info = self.data_infos[i]
            # TODO: set flag without width and height
            # if img_info['width'] / img_info['height'] > 1:
            self.flag[i] = 1
