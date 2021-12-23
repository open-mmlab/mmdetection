# Copyright (c) OpenMMLab. All rights reserved.
import copy
import csv
import json
import os.path as osp
import pickle
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmcv.utils import print_log

from mmdet.core import eval_map
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class OpenImagesDataset(CustomDataset):
    """Open Images dataset for detection."""

    def __init__(self,
                 label_description_file='',
                 get_parent_class=True,
                 hierarchy_file=None,
                 get_metas=True,
                 load_from_file=True,
                 meta_file='',
                 filter_labels=True,
                 **kwargs):
        """
        Args:
            label_description_file (str): File path to the label map proto.
            get_parent_class (bool): Whether get father class of the current
                class. Default: True.
            hierarchy_file (str): File path to the hierarchy for classes.
                Default: None.
            get_metas (bool): Whether get image metas. Default: True.
                The OpenImages annotations do not have image metas (width and
                height of the image), which will be used during evaluation.
                We provide two ways to get image metas in `OpenImagesDataset`:

                - 1. `load from file`: Load image metas from pkl file, which
                is suggested to use. We provided a script to get image metas:
                `tools/misc/get_image_metas.py`, which need to run this script
                 before training/testing. Please refer to
                `config/openimages/README.md` for more details.

                - 2. `load from pipeline`, which will get image metas during
                test time. However, this may reduce the inference speed,
                especially when using distribution.

            load_from_file (bool): Whether get image metas from pkl file.
            meta_file (str): File path to get image metas.
            filter_labels (bool): Whether filter unannotated classes.
                Default: True.
        """

        self.cat2label = defaultdict(str)
        self.index_dict = {}
        # need get `index_dict` before load annotations
        class_names = self.get_classes_from_csv(label_description_file)
        super(OpenImagesDataset, self).__init__(**kwargs)
        self.CLASSES = class_names
        if get_parent_class is True and hierarchy_file is not None:
            self.class_label_tree = self.get_father(hierarchy_file)
        self.get_parent_class = get_parent_class
        self.get_metas = get_metas
        self.load_from_file = load_from_file
        self.meta_file = meta_file
        if self.data_root is not None:
            if not osp.isabs(self.meta_file):
                self.meta_file = osp.join(self.data_root, self.meta_file)
        self.filter_labels = filter_labels
        self.rank, self.world_size = get_dist_info()
        self.temp_img_metas = []
        self.test_img_metas = []
        self.test_img_shapes = []
        self.load_from_pipeline = False if load_from_file else True

    def get_classes_from_csv(self, label_description_file):
        """Get class name and label map proto.

        Args:
            label_description_file (str): File path to the label map proto.

        Returns:
            list[str]: Class name of OpenImages.
        """

        index_list = []
        classes_names = []
        with open(label_description_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                self.cat2label[line[0]] = line[1]
                classes_names.append(line[1])
                index_list.append(line[0])
        self.index_dict = {index: i for i, index in enumerate(index_list)}
        assert len(self.cat2label) == len(self.index_dict) == \
               len(classes_names)
        return classes_names

    def list_from_csv(self, ann_file):
        """Load annotation from csv style ann_file.

        Args:
            ann_file (str): CSV style annotation file path.

        Returns:
            tuple: Returns (item_list, data_infos), where

            `item_list` (defaultdict[list[dict]]):
            Annotations where item of the defaultdict indicates an
            image, each of which has (n) dicts. Keys of dicts are:

                - `bbox` (list): of shape 4.
                - `label` (int): of shape 1.
                - `is_group_of` (bool): of shape 1.
                - `is_occluded` (bool): of shape 1.
                - `is_truncated` (bool): of shape 1.
                - `is_depiction` (bool): of shape 1.
                - `is_inside` (bool): of shape 1.
            `data_infos` (list[dict])： Data infos where each item of
            the list indicates an image. Keys of annotations are:

                - `img_id` (str): Image name.
                - `filename` (str): Image name with suffix.
        """

        item_list = defaultdict(list)
        data_infos = []
        cp_filename = None
        with open(ann_file, 'r') as f:
            reader = csv.reader(f)
            i = -1
            for line in reader:
                i += 1
                if i == 0:
                    continue
                else:
                    img_id = line[0]
                    filename = f'{img_id}.jpg'
                    label = int(self.index_dict[line[2]])
                    bbox = [
                        float(line[4]),  # xmin
                        float(line[6]),  # ymin
                        float(line[5]),  # xmax
                        float(line[7])  # ymax
                    ]
                    is_occluded = True if line[8] == 1 else False
                    is_truncated = True if line[9] == 1 else False
                    is_group_of = True if line[10] == 1 else False
                    is_depiction = True if line[11] == 1 else False
                    is_inside = True if line[12] == 1 else False

                    item_list[img_id].append(
                        dict(
                            bbox=bbox,
                            label=label,
                            is_occluded=is_occluded,
                            is_truncated=is_truncated,
                            is_group_of=is_group_of,
                            is_depiction=is_depiction,
                            is_inside=is_inside))
                    if filename != cp_filename:
                        data_infos.append(
                            dict(img_id=img_id, filename=filename))
                        cp_filename = filename
        return item_list, data_infos

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        self.ann_infos, data_infos = self.list_from_csv(ann_file)
        return data_infos

    def get_ann_info(self, idx):
        """Get OpenImages annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos[idx]['img_id']
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        is_occludeds = []
        is_truncateds = []
        is_group_ofs = []
        is_depictions = []
        is_insides = []
        for obj in self.ann_infos[img_id]:
            label = int(obj['label'])
            bbox = [
                float(obj['bbox'][0]),
                float(obj['bbox'][1]),
                float(obj['bbox'][2]),
                float(obj['bbox'][3])
            ]
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

    def get_meta_from_file(self, meta_file=''):
        """Get image metas from pkl file."""
        assert meta_file.endswith('pkl'), 'File name must be pkl suffix'
        print('load meta file begin')
        with open(meta_file, 'rb') as f:
            metas = pickle.load(f)
        assert len(metas) == len(self)
        for i in range(len(metas)):
            file_name = metas[i].data[0][0]['filename'].split('/')[-1]
            img_info = self.data_infos[i].get('img_info', None)
            if img_info is not None:
                assert file_name == img_info['filename'].split('/')[-1]
            else:
                assert file_name == self.data_infos[i]['filename']
            hw = metas[i].data[0][0]['ori_shape'][:2]
            self.test_img_shapes.append(hw)
        print('load meta file end')

    def get_meta_from_pipeline(self, results):
        """Get image metas from pipeline."""
        self.temp_img_metas.extend(results['img_metas'])
        if dist.is_available() and self.world_size > 1:
            from mmdet.apis.test import collect_results_cpu

            self.test_img_metas = collect_results_cpu(self.temp_img_metas,
                                                      len(self))
        else:
            self.test_img_metas = self.temp_img_metas

    def get_img_shape(self, metas):
        """Set images original shape into data_infos."""
        assert len(metas) == len(self)
        for i in range(len(metas)):
            file_name = metas[i].data['filename'].split('/')[-1]
            assert file_name == \
                   self.data_infos[i].get('filename', None).split('/')[-1]
            hw = metas[i].data['ori_shape'][:2]
            self.test_img_shapes.append(hw)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline."""
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)
        if self.get_metas and self.load_from_pipeline:
            self.get_meta_from_pipeline(results)
        return results

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn('OpenImageDatasets does not support '
                          'filtering empty gt images.')
        valid_inds = [i for i in range(len(self))]
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio."""
        self.flag = np.zeros(len(self), dtype=np.uint8)
        # TODO: set flag without width and height

    def get_father(self, hierarchy_file):
        """Get hierarchy for classes.

        Args:
            hierarchy_file (sty): File path to the hierarchy for classes.

        Returns:
            ndarray: The matrix of the corresponding relationship between
            the father class and the child class, of shape
            (class_num, class_num).
        """

        assert hierarchy_file.endswith('json')
        if self.data_root is not None:
            if not osp.isabs(hierarchy_file):
                hierarchy_file = osp.join(self.data_root, hierarchy_file)
        with open(hierarchy_file, 'r') as f:
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
        """Get matrix of the corresponding relationship between the father
        class and the child class.

        Args:
            hierarchy (dict): Including label name and corresponding
                subcategory. Keys of dicts are:

                - `LabeName` (str): Name of the label.
                - `Subcategory` (dict | list): Corresponding subcategory(ies).
            class_label_tree (ndarray): The matrix of the corresponding
                relationship between the father class and the child class,
                of shape (class_num, class_num).
            father (list): Corresponding father class.
            get_all_fathers (bool): Whether get all father name. Default: True

            Returns:
                ndarray: The matrix of the corresponding relationship between
                the father class and the child class, of shape
                (class_num, class_num).
        """

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
        """Add father classes of the corresponding class of the ground truth
        bboxes."""
        for i, ann in enumerate(annotations):
            assert len(ann['labels']) == len(ann['bboxes']) == \
                   len(ann['gt_is_group_ofs'])
            gt_bboxes = []
            gt_is_group_ofs = []
            gt_labels = []
            for j in range(len(ann['labels'])):
                label = ann['labels'][j]
                bbox = ann['bboxes'][j]
                is_group = ann['gt_is_group_ofs'][j]
                label = np.where(self.class_label_tree[label])[0]
                if len(label) > 1:
                    for k in range(len(label)):
                        gt_bboxes.append(bbox)
                        gt_is_group_ofs.append(is_group)
                        gt_labels.append(label[k])
                else:
                    gt_bboxes.append(bbox)
                    gt_is_group_ofs.append(is_group)
                    gt_labels.append(label[0])
            annotations[i] = dict(
                bboxes=np.array(gt_bboxes).astype(np.float32),
                labels=np.array(gt_labels).astype(np.int64),
                bboxes_ignore=ann['bboxes_ignore'],
                gt_is_group_ofs=np.array(gt_is_group_ofs).astype(np.bool))

        return annotations

    def get_father_results(self, det_results, annotations):
        """Add father classes of the corresponding class of the detection
        bboxes."""
        for i in range(len(det_results)):
            results = copy.deepcopy(det_results[i])
            valid_classes = np.where(
                np.array([[bbox.shape[0]] for bbox in det_results[i]]) != 0)[
                    0]  # label begin from 1
            allowed_labeles = np.unique(annotations[i]['labels'])

            for valid_class in valid_classes:
                det_cls = np.where(self.class_label_tree[valid_class])[0]
                for index in det_cls:
                    if index in allowed_labeles and index != valid_class:
                        det_results[i][index] = \
                            np.concatenate((det_results[i][index],
                                            results[valid_class]))
                    elif index not in allowed_labeles and self.filter_labels:
                        # Remove useless parts
                        det_results[i][index] = np.empty(
                            (0, 5)).astype(np.float32)
        return det_results

    def denormalize_gt_bboxes(self, annotations):
        """Convert ground truth bboxes from relative position to absolute
        position."""
        assert len(self.test_img_shapes) == len(annotations)
        for i in range(len(annotations)):
            h, w = self.test_img_shapes[i]
            annotations[i]['bboxes'][:, 0::2] *= w
            annotations[i]['bboxes'][:, 1::2] *= h
        return annotations

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 iou_thr=0.5,
                 ioa_thr=0.5,
                 scale_ranges=None,
                 denorm_gt_bbox=True,
                 use_group_of=True):
        """Evaluate in OpenImages.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Option is
                 'mAP'. Default: 'mAP'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            ioa_thr (float | list[float]): IoA threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None
            denorm_gt_bbox (bool): Whether to denorm ground truth bboxes from
                relative position to absolute position. Default: True
            use_group_of (bool): Whether consider group of groud truth bboxes
                during evaluating. Default: True.

        Returns:
            dict[str, float]: AP metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]

        # load from file
        if self.get_metas and self.load_from_file:
            self.get_meta_from_file(self.meta_file)
        # load from pipeline
        else:
            self.get_img_shape(self.test_img_metas)

        if len(self.test_img_shapes) > len(self):
            self.test_img_shapes = self.test_img_shapes[:len(self)]

        if denorm_gt_bbox:
            annotations = self.denormalize_gt_bboxes(annotations)

        # Reset test_image_metas, temp_image_metas and test_img_shapes
        # to avoid potential error
        self.temp_img_metas = []
        self.test_img_shapes = []
        self.test_img_metas = []
        if self.get_parent_class:
            annotations = self.get_gt_fathers(annotations)
            results = self.get_father_results(results, annotations)

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
                    logger=logger,
                    use_group_of=use_group_of)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        return eval_results


@DATASETS.register_module()
class OpenImagesChallengeDataset(OpenImagesDataset):
    """Open Images Challenge dataset for detection."""

    def __init__(self, **kwargs):
        super(OpenImagesChallengeDataset, self).__init__(**kwargs)

    def get_classes_from_csv(self, label_description_file):
        """Get class name and label map proto.

        Args:
            label_description_file (str): File path to the label map proto.

        Returns:
            list: Class name of OpenImages.
        """

        label_list = []
        id_list = []
        with open(label_description_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                label_list.append(line[1])
                id_list.append(int(line[2]))
        indexes = np.argsort(id_list)
        assert len(label_list) == len(id_list)
        classes_names = []
        for index in indexes:
            classes_names.append(label_list[index])
        return classes_names

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        assert ann_file.endswith('txt')
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
                labels.append(int(sp[0]) - 1)  # labels begin from 1
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

        return ann_infos

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline."""
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
        """Get testing data after pipeline."""
        ann_info = self.data_infos[idx]
        results = dict(img_info=ann_info['img_info'])
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)

        results = self.pipeline(results)
        if self.get_metas and self.load_from_pipeline:
            self.get_meta_from_pipeline(results)
        return results

    def get_father(self, hierarchy_file):
        """Get hierarchy for classes.

        Args:
            hierarchy_file (str): File path to the hierarchy for classes.

        Returns:
            ndarray: The matrix of the corresponding
            relationship between the father class and the child class,
            of shape (class_num, class_num).
        """

        assert hierarchy_file.endswith('np')
        class_label_tree = np.load(hierarchy_file, allow_pickle=True)
        return class_label_tree[1:, 1:]

    def get_ann_info(self, idx):
        """Get OpenImages annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.data_infos[idx]['ann_info']
