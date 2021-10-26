import copy
import csv
import os.path as osp
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
from mmcv.utils import print_log

from mmdet.core import eval_map
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
                is_occluded = True if line[8] == 1 else False
                is_truncated = True if line[9] == 1 else False
                is_group_of = True if line[10] == 1 else False
                is_depiction = True if line[11] == 1 else False
                is_inside = True if line[12] == 1 else False
                # ann[img_id].append()
                item_list[img_id].append(
                    dict(
                        bbox=[x_min, x_max, y_mix, y_max],
                        label=label,
                        IsOccluded=is_occluded,
                        IsTruncated=is_truncated,
                        IsGroupOf=is_group_of,
                        IsDepiction=is_depiction,
                        IsInside=is_inside))
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

    def __init__(self,
                 label_csv_path='',
                 min_size=None,
                 need_get_father=True,
                 class_label_tree_path=None,
                 **kwargs):
        super(OpenImagesChallengeDataset, self).__init__(**kwargs)
        self.min_size = min_size
        self.CLASSES = self.get_classes_from_csv(label_csv_path)
        if need_get_father is True and class_label_tree_path is not None:
            self.class_label_tree = self.get_father(class_label_tree_path)
        self.need_get_father = need_get_father
        self.test_img_shape = [[] for _ in range(len(self))]

    def get_classes_from_csv(self, label_csv_path):
        """class-descriptions-boxable.csv."""
        label_list = []
        id_list = []
        # self.cat2label = {}
        if self.data_root is not None:
            label_csv_path = osp.join(self.data_root, label_csv_path)
        with open(label_csv_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                label_list.append(line[1])
                id_list.append(int(line[2]))
        indexes = np.argsort(id_list)
        assert len(label_list) == len(id_list)
        classes = []
        for index in indexes:
            classes.append(label_list[index])

        return classes

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
                # OpenImage labels begin from 1
                labels.append(int(sp[0]) - 1)
                is_group_ofs.append(True if int(sp[5]) == 1 else False)
            i += img_gt_size

            gt_bboxes = np.array(bboxes, dtype=np.float32)
            gt_labels = np.array(labels, dtype=np.int64)
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
            gt_is_group_ofs = np.array(is_group_ofs, dtype=np.bool)

            img_info = dict(filename=filename)
            ann_info = dict(
                bboxes=gt_bboxes,
                labels=gt_labels,
                bboxes_ignore=gt_bboxes_ignore,
                gt_is_group_ofs=gt_is_group_ofs)
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

        results = self.pipeline(results)
        img_metas = results['img_metas'][0]
        self.test_img_shape[idx] = img_metas.data['ori_shape'][0:2]
        return results

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

    def get_father(self, class_label_tree_path):
        class_label_tree = np.load(class_label_tree_path, allow_pickle=True)
        return class_label_tree

    def get_gt_fathers(self, annotations):
        for i, ann in enumerate(annotations):
            assert len(ann['labels']) == len(ann['bboxes']) == \
                   len(ann['gt_is_group_ofs'])
            gt_bboxes = []
            gt_is_group_ofs = []
            gt_labels = []
            for j in range(len(ann['labels'])):
                label = ann['labels'][j] + 1
                bbox = ann['bboxes'][j]
                is_group = ann['gt_is_group_ofs'][j]
                label = np.where(self.class_label_tree[label])[0]
                if len(label) > 1:
                    for k in range(len(label)):
                        gt_bboxes.append(bbox)
                        gt_is_group_ofs.append(is_group)
                        gt_labels.append(label[k] - 1)
                else:
                    gt_bboxes.append(bbox)
                    gt_is_group_ofs.append(is_group)
                    gt_labels.append(label[0] - 1)
            annotations[i] = dict(
                bboxes=np.array(gt_bboxes).astype(np.float32),
                labels=np.array(gt_labels).astype(np.int64),
                bboxes_ignore=ann['bboxes_ignore'],
                gt_is_group_ofs=np.array(gt_is_group_ofs).astype(np.bool))

        return annotations

    def get_result_fathers(self, det_results, annotations):
        for i in range(len(det_results)):
            print(f'{i + 1}/{len(det_results)}')
            results = copy.deepcopy(det_results[i])
            valid_classes = np.where(
                np.array([[bbox.shape[0]] for bbox in det_results[i]]) != 0
            )[0] + 1  # openimage label begin from 1
            labels = annotations[i]['labels'] + 1
            for valid_class in valid_classes:
                det_cls = np.where(self.class_label_tree[valid_class])[0]
                for index in det_cls:
                    if index in labels and index != valid_class:
                        det_results[i][index - 1] = \
                            np.concatenate((det_results[i][index - 1],
                                            results[valid_class - 1]))
                    elif index not in labels:
                        # Remove useless parts
                        det_results[i][index - 1] = np.empty(
                            (0, 5)).astype(np.float32)
        return det_results

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.data_infos[idx]['ann_info']

    def normed_bbox(self, annotations):
        for i, ann in enumerate(annotations):
            h, w = self.test_img_shape[i]
            annotations[i]['bboxes'][:, 0] *= w
            annotations[i]['bboxes'][:, 1] *= h
            annotations[i]['bboxes'][:, 2] *= w
            annotations[i]['bboxes'][:, 3] *= h
        return annotations

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 ioa_thr=0.5,
                 scale_ranges=None,
                 normed_bbox=True):

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        if normed_bbox:
            annotations = self.normed_bbox(annotations)
        if self.need_get_father:
            annotations = self.get_gt_fathers(annotations)
            results = self.get_result_fathers(results, annotations)

        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        ioa_thrs = [ioa_thr] if isinstance(ioa_thr, float) or ioa_thr is None \
            else ioa_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list) and isinstance(ioa_thrs, list)
            assert len(ioa_thrs) == len(iou_thrs)
            mean_aps = []
            for iou_thr, ioa_thr in zip(iou_thrs, ioa_thrs):
                print_log(f'\n{"-" * 15}iou_thr, ioa_thr: {iou_thr}, {ioa_thr}'
                          f'{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    ioa_thr=ioa_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        return eval_results
