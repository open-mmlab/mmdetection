import copy
import csv
import json
import os.path as osp
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
from mmcv.utils import print_log

from mmdet.core import eval_map
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class OpenImagesDataset(CustomDataset):

    def __init__(self,
                 label_csv_path='',
                 min_size=None,
                 need_get_father=True,
                 hierarchy_file_path=None,
                 **kwargs):
        super(OpenImagesDataset, self).__init__(**kwargs)
        self.min_size = min_size
        self.cat2label = defaultdict(str)
        self.index_dict = {}
        self.CLASSES = self.get_classes_from_csv(label_csv_path)
        if need_get_father is True and hierarchy_file_path is not None:
            self.class_label_tree = self.get_father(hierarchy_file_path)
        self.need_get_father = need_get_father
        self.test_img_shape = [[] for _ in range(len(self))]

    def get_classes_from_csv(self, label_csv_path):
        """class-descriptions-boxable.csv."""
        index_list = []
        classes = []
        with open(label_csv_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                self.cat2label[line[0]] = line[1]
                classes.append(line[1])
                index_list.append(line[0])
        self.index_dict = {index: i for i, index in enumerate(index_list)}
        assert len(self.cat2label) == len(self.index_list) == len(classes)
        return classes

    def list_from_csv(self, ann_file):
        """Load annotation from csv style ann_file.

        Args:
            ann_file (str): .

        Returns:
            dict[list[dict]]: Annotation info from csv file.
        """

        item_list = defaultdict(list)
        data_infos = []
        with open(ann_file, 'r') as f:
            reader = csv.reader(f)
            i = -1
            for line in reader:
                i += 1
                if i == 0:
                    continue
                # else:
                elif i < 30:  # for debug, TODO: delete later
                    filename = line[0]
                    label = int(self.index_dict[line[2]])
                    bbox = [
                        float(line[4]),  # xmin
                        float(line[6]),  # ymin
                        float(line[5]),  # xmax
                        float(line[7])
                    ]  # ymax
                    is_occluded = True if line[8] == 1 else False
                    is_truncated = True if line[9] == 1 else False
                    is_group_of = True if line[10] == 1 else False
                    is_depiction = True if line[11] == 1 else False
                    is_inside = True if line[12] == 1 else False

                    item_list[filename].append(
                        dict(
                            bbox=bbox,
                            label=label,
                            is_occluded=is_occluded,
                            is_truncated=is_truncated,
                            is_group_of=is_group_of,
                            is_depiction=is_depiction,
                            is_inside=is_inside))
                    data_infos.append(dict(filename=filename))

        return item_list, data_infos

    def load_annotations(self, ann_file):
        self.ann_infos, data_infos = self.list_from_csv(ann_file)
        return data_infos

    def get_ann_info(self, idx):
        filename = self.data_infos[idx]['filename']
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        is_occludeds = []
        is_truncateds = []
        is_group_ofs = []
        is_depictions = []
        is_insides = []
        for obj in self.ann_infos[filename]:
            label = int(obj['label'])
            bbox = [
                int(float(obj['bbox'][0])),
                int(float(obj['bbox'][1])),
                int(float(obj['bbox'][2])),
                int(float(obj['bbox'][3]))
            ]
            ignore = False
            # this is a hard code because we cannot get wh in OpenImages
            # Should we delete this?
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

            # Other parameters
            is_occludeds.append(obj['is_occluded'])
            is_truncateds.append(obj['is_truncated'])
            is_group_ofs.append(obj['is_group_of'])
            is_depictions.append(obj['is_depiction'])
            is_insides.append(obj['is_inside'])
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore)
            labels_ignore = np.array(labels_ignore)

        assert len(is_group_ofs) == len(labels) == len(bboxes)
        gt_is_group_ofs = np.array(is_group_ofs, dtype=np.bool)

        # These parameters is not used yet.
        is_occludeds = np.array(is_occludeds, dtype=np.bool)
        is_truncateds = np.array(is_truncateds, dtype=np.bool)
        is_depictions = np.array(is_depictions, dtype=np.bool)
        is_insides = np.array(is_insides, dtype=np.bool)

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64),
            gt_is_group_ofs=gt_is_group_ofs,
            is_occludeds=is_occludeds,
            is_truncateds=is_truncateds,
            is_depictions=is_depictions,
            is_insides=is_insides)

        return ann

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
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

    def get_father(self, hierarchy_file_path):
        """

        Args:
            json_path:

        Returns:
            class_label_tree
        """
        assert hierarchy_file_path.endswith('json')
        if self.data_root is not None:
            if not osp.isabs(hierarchy_file_path):
                hierarchy_file_path = osp.join(self.data_root,
                                               hierarchy_file_path)
        with open(hierarchy_file_path, 'r') as f:
            hierarchy = json.load(f)
        class_num = len(self.CLASSES)
        class_label_tree = np.eye(class_num, class_num)
        class_label_tree = self.get_parent_children(hierarchy,
                                                    class_label_tree)
        return class_label_tree

    def get_parent_children(self,
                            hierarchy,
                            class_label_tree,
                            father=[],
                            get_all_fathers=True):
        if 'Subcategory' in hierarchy:
            for node in hierarchy['Subcategory']:
                if 'LabelName' in node:
                    children_name = node['LabelName']
                    children_index = self.index_dict[children_name]
                    children = [children_index]
                else:
                    continue
                if len(father) > 0:
                    for father_index in father:
                        if get_all_fathers:
                            children.append(father_index)
                        class_label_tree[children_index, father_index] = 1

                class_label_tree = self.get_parent_children(
                    node, class_label_tree, father=children)

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

    def normed_bbox(self, annotations):
        """

        Args:
            annotations:

        Returns:

        """
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


#
@DATASETS.register_module()
class OpenImagesChallengeDataset(OpenImagesDataset):
    """"""

    def __init__(self, **kwargs):
        super(OpenImagesChallengeDataset, self).__init__(**kwargs)
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

    def get_father(self, hierarchy_file_path):
        assert hierarchy_file_path.endswith('np')
        class_label_tree = np.load(hierarchy_file_path, allow_pickle=True)
        return class_label_tree

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.data_infos[idx]['ann_info']
