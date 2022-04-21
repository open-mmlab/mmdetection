# Copyright (c) OpenMMLab. All rights reserved.
import copy
import csv
import json
import os.path as osp
import warnings
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmcv.utils import print_log

from mmdet.core import eval_map
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class OpenImagesDataset(CustomDataset):
    """Open Images dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        label_file (str): File path of the label description file that
            maps the classes names in MID format to their short
            descriptions.
        image_level_ann_file (str): Image level annotation, which is used
            in evaluation.
        get_supercategory (bool): Whether to get parent class of the
            current class. Default: True.
        hierarchy_file (str): The file path of the class hierarchy.
            Default: None.
        get_metas (bool): Whether to get image metas in testing or
            validation time. This should be `True` during evaluation.
            Default: True. The OpenImages annotations do not have image
            metas (width and height of the image), which will be used
            during evaluation. We provide two ways to get image metas
            in `OpenImagesDataset`:

            - 1. `load from file`: Load image metas from pkl file, which
              is suggested to use. We provided a script to get image metas:
              `tools/misc/get_image_metas.py`, which need to run
              this script before training/testing. Please refer to
              `config/openimages/README.md` for more details.

            - 2. `load from pipeline`, which will get image metas during
              test time. However, this may reduce the inference speed,
              especially when using distribution.

        load_from_file (bool): Whether to get image metas from pkl file.
        meta_file (str): File path to get image metas.
        filter_labels (bool): Whether filter unannotated classes.
            Default: True.
        load_image_level_labels (bool): Whether load and consider image
            level labels during evaluation. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 ann_file,
                 label_file='',
                 image_level_ann_file='',
                 get_supercategory=True,
                 hierarchy_file=None,
                 get_metas=True,
                 load_from_file=True,
                 meta_file='',
                 filter_labels=True,
                 load_image_level_labels=True,
                 file_client_args=dict(backend='disk'),
                 **kwargs):
        # may get error if use other file_client
        self.file_client_args = file_client_args

        self.cat2label = defaultdict(str)
        self.index_dict = {}

        # Although it will init file_client in `CustomDataset`,
        # it needs to be init here.
        file_client = mmcv.FileClient(**file_client_args)
        # need get `index_dict` before load annotations
        assert label_file.endswith('csv')
        if hasattr(file_client, 'get_local_path'):
            with file_client.get_local_path(label_file) as local_path:
                class_names = self.get_classes_from_csv(local_path)
        else:
            class_names = self.get_classes_from_csv(label_file)
        super(OpenImagesDataset, self).__init__(
            ann_file=ann_file, file_client_args=file_client_args, **kwargs)
        self.CLASSES = class_names
        self.image_level_ann_file = image_level_ann_file
        self.load_image_level_labels = load_image_level_labels
        if get_supercategory is True:
            assert hierarchy_file is not None
            if self.__class__.__name__ == 'OpenImagesDataset':
                assert hierarchy_file.endswith('json')
            elif self.__class__.__name__ == 'OpenImagesChallengeDataset':
                assert hierarchy_file.endswith('np')
            else:
                raise NotImplementedError
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(
                        hierarchy_file) as local_path:
                    self.class_label_tree = self.get_relation_matrix(
                        local_path)
            else:
                self.class_label_tree = self.get_relation_matrix(
                    hierarchy_file)
        self.get_supercategory = get_supercategory
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

    def get_classes_from_csv(self, label_file):
        """Get classes name from file.

        Args:
            label_file (str): File path of the label description file that
                maps the classes names in MID format to their short
                descriptions.

        Returns:
            list[str]: Class name of OpenImages.
        """

        index_list = []
        classes_names = []
        with open(label_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                self.cat2label[line[0]] = line[1]
                classes_names.append(line[1])
                index_list.append(line[0])
        self.index_dict = {index: i for i, index in enumerate(index_list)}
        return classes_names

    def load_annotations(self, ann_file):
        """Load annotation from annotation file.

        Special described `self.data_infos` (defaultdict[list[dict]])
        in this function: Annotations where item of the defaultdict
        indicates an image, each of which has (n) dicts. Keys of dicts are:

            - `bbox` (list): coordinates of the box, in normalized image
              coordinates, of shape 4.
            - `label` (int): the label id.
            - `is_group_of` (bool):  Indicates that the box spans a group
              of objects (e.g., a bed of flowers or a crowd of people).
            - `is_occluded` (bool): Indicates that the object is occluded
              by another object in the image.
            - `is_truncated` (bool): Indicates that the object extends
              beyond the boundary of the image.
            - `is_depiction` (bool): Indicates that the object is a
              depiction.
            - `is_inside` (bool): Indicates a picture taken from the
              inside of the object.

        Args:
            ann_file (str): CSV style annotation file path.

        Returns:
            list[dict]:  Data infos where each item of the list
            indicates an image. Keys of annotations are:

                - `img_id` (str): Image name.
                - `filename` (str): Image name with suffix.
        """
        self.ann_infos = defaultdict(list)
        data_infos = []
        cp_filename = None
        with open(ann_file, 'r') as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                img_id = line[0]
                filename = f'{img_id}.jpg'
                label_id = line[2]
                assert label_id in self.index_dict
                label = int(self.index_dict[label_id])
                bbox = [
                    float(line[4]),  # xmin
                    float(line[6]),  # ymin
                    float(line[5]),  # xmax
                    float(line[7])  # ymax
                ]
                is_occluded = True if int(line[8]) == 1 else False
                is_truncated = True if int(line[9]) == 1 else False
                is_group_of = True if int(line[10]) == 1 else False
                is_depiction = True if int(line[11]) == 1 else False
                is_inside = True if int(line[12]) == 1 else False

                self.ann_infos[img_id].append(
                    dict(
                        bbox=bbox,
                        label=label,
                        is_occluded=is_occluded,
                        is_truncated=is_truncated,
                        is_group_of=is_group_of,
                        is_depiction=is_depiction,
                        is_inside=is_inside))
                if filename != cp_filename:
                    data_infos.append(dict(img_id=img_id, filename=filename))
                    cp_filename = filename
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
        metas = mmcv.load(
            meta_file,
            file_format='pkl',
            file_client_args=self.file_client_args)
        assert len(metas) == len(self)
        for i in range(len(metas)):
            file_name = osp.split(metas[i]['filename'])[-1]
            img_info = self.data_infos[i].get('img_info', None)
            if img_info is not None:
                assert file_name == osp.split(img_info['filename'])[-1]
            else:
                assert file_name == self.data_infos[i]['filename']
            hw = metas[i]['ori_shape'][:2]
            self.test_img_shapes.append(hw)

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
            file_name = osp.split(metas[i].data['ori_filename'])[-1]
            img_info = self.data_infos[i].get('img_info', None)
            if img_info is not None:
                assert file_name == osp.split(img_info['filename'])[-1]
            else:
                assert file_name == self.data_infos[i]['filename']
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

    def get_relation_matrix(self, hierarchy_file):
        """Get hierarchy for classes.

        Args:
            hierarchy_file (sty): File path to the hierarchy for classes.

        Returns:
            ndarray: The matrix of the corresponding relationship between
            the parent class and the child class, of shape
            (class_num, class_num).
        """

        if self.data_root is not None:
            if not osp.isabs(hierarchy_file):
                hierarchy_file = osp.join(self.data_root, hierarchy_file)
        with open(hierarchy_file, 'r') as f:
            hierarchy = json.load(f)
        class_num = len(self.CLASSES)
        class_label_tree = np.eye(class_num, class_num)
        class_label_tree = self._convert_hierarchy_tree(
            hierarchy, class_label_tree)
        return class_label_tree

    def _convert_hierarchy_tree(self,
                                hierarchy_map,
                                class_label_tree,
                                parents=[],
                                get_all_parents=True):
        """Get matrix of the corresponding relationship between the parent
        class and the child class.

        Args:
            hierarchy_map (dict): Including label name and corresponding
                subcategory. Keys of dicts are:

                - `LabeName` (str): Name of the label.
                - `Subcategory` (dict | list): Corresponding subcategory(ies).
            class_label_tree (ndarray): The matrix of the corresponding
                relationship between the parent class and the child class,
                of shape (class_num, class_num).
            parents (list): Corresponding parent class.
            get_all_parents (bool): Whether get all parent names.
                Default: True

        Returns:
            ndarray: The matrix of the corresponding relationship between
            the parent class and the child class, of shape
            (class_num, class_num).
        """

        if 'Subcategory' in hierarchy_map:
            for node in hierarchy_map['Subcategory']:
                if 'LabelName' in node:
                    children_name = node['LabelName']
                    children_index = self.index_dict[children_name]
                    children = [children_index]
                else:
                    continue
                if len(parents) > 0:
                    for parent_index in parents:
                        if get_all_parents:
                            children.append(parent_index)
                        class_label_tree[children_index, parent_index] = 1

                class_label_tree = self._convert_hierarchy_tree(
                    node, class_label_tree, parents=children)

        return class_label_tree

    def add_supercategory_ann(self, annotations):
        """Add parent classes of the corresponding class of the ground truth
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

    def process_results(self, det_results, annotations,
                        image_level_annotations):
        """Process results of the corresponding class of the detection bboxes.

        Note: It will choose to do the following two processing according to
        the parameters:

        1. Whether to add parent classes of the corresponding class of the
        detection bboxes.

        2. Whether to ignore the classes that unannotated on that image.
        """
        if image_level_annotations is not None:
            assert len(annotations) == \
                   len(image_level_annotations) == \
                   len(det_results)
        else:
            assert len(annotations) == len(det_results)
        for i in range(len(det_results)):
            results = copy.deepcopy(det_results[i])
            valid_classes = np.where(
                np.array([[bbox.shape[0]] for bbox in det_results[i]]) != 0)[0]
            if image_level_annotations is not None:
                labels = annotations[i]['labels']
                image_level_labels = \
                    image_level_annotations[i]['image_level_labels']
                allowed_labeles = np.unique(
                    np.append(labels, image_level_labels))
            else:
                allowed_labeles = np.unique(annotations[i]['labels'])

            for valid_class in valid_classes:
                det_cls = np.where(self.class_label_tree[valid_class])[0]
                for index in det_cls:
                    if index in allowed_labeles and \
                            index != valid_class and \
                            self.get_supercategory:
                        det_results[i][index] = \
                            np.concatenate((det_results[i][index],
                                            results[valid_class]))
                    elif index not in allowed_labeles and self.filter_labels:
                        # Remove useless parts
                        det_results[i][index] = np.empty(
                            (0, 5)).astype(np.float32)
        return det_results

    def load_image_label_from_csv(self, image_level_ann_file):
        """Load image level annotations from csv style ann_file.

        Args:
            image_level_ann_file (str): CSV style image level annotation
                file path.

        Returns:
            defaultdict[list[dict]]: Annotations where item of the defaultdict
            indicates an image, each of which has (n) dicts.
            Keys of dicts are:

                - `image_level_label` (int): Label id.
                - `confidence` (float): Labels that are human-verified to be
                  present in an image have confidence = 1 (positive labels).
                  Labels that are human-verified to be absent from an image
                  have confidence = 0 (negative labels). Machine-generated
                  labels have fractional confidences, generally >= 0.5.
                  The higher the confidence, the smaller the chance for
                  the label to be a false positive.
        """

        item_lists = defaultdict(list)
        with open(image_level_ann_file, 'r') as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                img_id = line[0]
                item_lists[img_id].append(
                    dict(
                        image_level_label=int(self.index_dict[line[2]]),
                        confidence=float(line[3])))
        return item_lists

    def get_image_level_ann(self, image_level_ann_file):
        """Get OpenImages annotation by index.

        Args:
            image_level_ann_file (str): CSV style image level annotation
                file path.

        Returns:
            dict: Annotation info of specified index.
        """

        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(image_level_ann_file) \
                    as local_path:
                item_lists = self.load_image_label_from_csv(local_path)
        else:
            item_lists = self.load_image_label_from_csv(image_level_ann_file)
        image_level_annotations = []
        for i in range(len(self)):
            img_info = self.data_infos[i].get('img_info', None)
            if img_info is not None:
                # for Open Images Challenges
                img_id = osp.split(img_info['filename'])[-1][:-4]
            else:
                # for Open Images v6
                img_id = self.data_infos[i]['img_id']
            item_list = item_lists.get(img_id, None)
            if item_list is not None:
                image_level_labels = []
                confidences = []
                for obj in item_list:
                    image_level_label = int(obj['image_level_label'])
                    confidence = float(obj['confidence'])

                    image_level_labels.append(image_level_label)
                    confidences.append(confidence)

                if not image_level_labels:
                    image_level_labels = np.zeros((0, ))
                    confidences = np.zeros((0, ))
                else:
                    image_level_labels = np.array(image_level_labels)
                    confidences = np.array(confidences)
            else:
                image_level_labels = np.zeros((0, ))
                confidences = np.zeros((0, ))
            ann = dict(
                image_level_labels=image_level_labels.astype(np.int64),
                confidences=confidences.astype(np.float32))
            image_level_annotations.append(ann)

        return image_level_annotations

    def denormalize_gt_bboxes(self, annotations):
        """Convert ground truth bboxes from relative position to absolute
        position.

        Only used in evaluating time.
        """
        assert len(self.test_img_shapes) == len(annotations)
        for i in range(len(annotations)):
            h, w = self.test_img_shapes[i]
            annotations[i]['bboxes'][:, 0::2] *= w
            annotations[i]['bboxes'][:, 1::2] *= h
        return annotations

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        return self.get_ann_info(idx)['labels'].astype(np.int).tolist()

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

        if self.load_image_level_labels:
            image_level_annotations = \
                self.get_image_level_ann(self.image_level_ann_file)
        else:
            image_level_annotations = None

        # load metas from file
        if self.get_metas and self.load_from_file:
            assert self.meta_file.endswith(
                'pkl'), 'File name must be pkl suffix'
            self.get_meta_from_file(self.meta_file)
        # load metas from pipeline
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
        if self.get_supercategory:
            annotations = self.add_supercategory_ann(annotations)

        results = self.process_results(results, annotations,
                                       image_level_annotations)
        if use_group_of:
            assert ioa_thr is not None, \
                'ioa_thr must have value when using group_of in evaluation.'

        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        ioa_thrs = [ioa_thr] if isinstance(ioa_thr, float) or ioa_thr is None \
            else ioa_thr

        # get dataset type
        if len(self.CLASSES) == 500:
            ds_name = 'oid_challenge'
        elif len(self.CLASSES) == 601:
            ds_name = 'oid_v6'
        else:
            ds_name = self.CLASSES
            warnings.warn('Cannot infer dataset type from the length of the '
                          'classes. Set `oid_v6` as dataset type.')

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
                    dataset=ds_name,
                    logger=logger,
                    use_group_of=use_group_of)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        return eval_results


@DATASETS.register_module()
class OpenImagesChallengeDataset(OpenImagesDataset):
    """Open Images Challenge dataset for detection."""

    def __init__(self, ann_file, **kwargs):
        assert ann_file.endswith('txt')
        super(OpenImagesChallengeDataset, self).__init__(
            ann_file=ann_file, **kwargs)

    def get_classes_from_csv(self, label_file):
        """Get classes name from file.

        Args:
            label_file (str): File path of the label description file that
                maps the classes names in MID format to their short
                descriptions.

        Returns:
            list: Class name of OpenImages.
        """

        label_list = []
        id_list = []
        with open(label_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                label_name = line[0]
                label_id = int(line[2])

                label_list.append(line[1])
                id_list.append(label_id)
                self.index_dict[label_name] = label_id - 1

        indexes = np.argsort(id_list)
        classes_names = []
        for index in indexes:
            classes_names.append(label_list[index])
        return classes_names

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
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

    def get_relation_matrix(self, hierarchy_file):
        """Get hierarchy for classes.

        Args:
            hierarchy_file (str): File path to the hierarchy for classes.

        Returns:
            ndarray: The matrix of the corresponding
            relationship between the parent class and the child class,
            of shape (class_num, class_num).
        """
        class_label_tree = np.load(hierarchy_file, allow_pickle=True)
        return class_label_tree[1:, 1:]

    def get_ann_info(self, idx):
        """Get OpenImages annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        # avoid some potential error
        data_infos = copy.deepcopy(self.data_infos[idx]['ann_info'])
        return data_infos

    def load_image_label_from_csv(self, image_level_ann_file):
        """Load image level annotations from csv style ann_file.

        Args:
            image_level_ann_file (str): CSV style image level annotation
                file path.

        Returns:
            defaultdict[list[dict]]: Annotations where item of the defaultdict
            indicates an image, each of which has (n) dicts.
            Keys of dicts are:

                - `image_level_label` (int): of shape 1.
                - `confidence` (float): of shape 1.
        """

        item_lists = defaultdict(list)
        with open(image_level_ann_file, 'r') as f:
            reader = csv.reader(f)
            i = -1
            for line in reader:
                i += 1
                if i == 0:
                    continue
                else:
                    img_id = line[0]
                    label_id = line[1]
                    assert label_id in self.index_dict
                    image_level_label = int(self.index_dict[label_id])
                    confidence = float(line[2])
                    item_lists[img_id].append(
                        dict(
                            image_level_label=image_level_label,
                            confidence=confidence))
        return item_lists
