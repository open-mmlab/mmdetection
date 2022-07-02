# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class CustomDataset(Dataset):
    """用于检测的自定义数据集.

    标注文件格式如下所示。 `ann` 字段对于验证阶段是可有可无的.

    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]

    Args:
        ann_file (str): 标注文件路径.
        pipeline (list[dict]): 数据处理管道.
        classes (str | Sequence[str], optional): 指定要加载的类.如果是 None,将使用 ``cls.CLASSES``.默认值:无.
        data_root (str, optional): 如果指定, ``ann_file``,``img_prefix``, ``seg_prefix``, ``proposal_file``的数据根目录,
        test_mode (bool, optional): 如果为 True,则不会加载标注文件.
        filter_empty_gt (bool, optional): 如果设置为 true,则将过滤掉没有标注框的图像.
                                          此选项仅在 `test_mode=False` 时有效,即我们在测试期间从不过滤图像.
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)
        self.CLASSES = self.get_classes(classes)

        # 如果指定了 data_root 并且 下列各路径并非绝对路径时.则使data_root成为下列路径的父目录
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root, self.proposal_file)
        # 加载标注文件 (and proposals) TODO proposals?
        if hasattr(self.file_client, 'get_local_path'):
            # 如果 self.ann_file 是非本地路径则get_local_path方法会将其上面的数据下载到本地local_path,如果是那么直接返回local_path
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(local_path)
        else:
            warnings.warn(f'正在使用的 MMCV 版本没有 get_local_path方法. 我们将 {self.ann_file} 视为本地路径,'
                          '如果它不是本地路径，它可能会导致错误.如果遇到错误请使用 MMCV>= 1.3.16的版本.')
            self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(self.proposal_file) as local_path:
                    self.proposals = self.load_proposals(local_path)
            else:
                warnings.warn(f'正在使用的 MMCV 版本没有 get_local_path方法. 我们将 {self.ann_file} 视为本地路径,'
                              '如果它不是本地路径，它可能会导致错误.如果遇到错误请使用 MMCV>= 1.3.16的版本.')
                self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # 过滤宽或高小于指定尺寸且没有标注的图像(仅在训练阶段应用)
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # 为采样数据设置组flag值
            self._set_group_flag()

        # 初始化数据处理管道
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """数据样本总数."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """从标注文件加载标注信息."""
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        """Load proposal from proposal file."""
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        """根据索引获取标注信息中的ann字段.

        Args:
            idx (int): 指定索引.

        Returns:
            dict: 指定索引的ann字段信息.
        """

        return self.data_infos[idx]['ann']

    def get_cat_ids(self, idx):
        """根据索引获取类别id.

        Args:
            idx (int): 指定索引.

        Returns:
            list[int]: 指定索引的图片中的所有类别id.
        """

        return self.data_infos[idx]['ann']['labels'].astype(np.int).tolist()

    def pre_pipeline(self, results):
        """为数据管道准备字典格式的数据."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """过滤宽或高小于指定尺寸太小的图像."""
        if self.filter_empty_gt:
            warnings.warn('CustomDataset基础类 不支持过滤空标签图像.')
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """根据图像宽高比设置flag.宽高比大于 1 的图像的flag值为 1,否则为 0."""
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """从与给定索引相同的flag值中获取另一个随机索引."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """获取经过数据增强管道后的 训练/测试 数据.

        Args:
            idx (int): 指定索引.

        Returns:
            dict: 训练/测试数据 (如果 `test_mode` 设置为 True，则带有标注信息).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """获取经过数据增强后的训练数据与标注信息.

        Args:
            idx (int): 指定索引.

        Returns:
            dict: 增加新字段的标注信息.
        results = {
        'img_info': {'file_name': '000921.jpg', 'id': 754, 'width': 800, 'height': 600, 'filename': '000921.jpg'},
        'ann_info': {'bboxes': array([[108., 350., 149., 455.],[ 68., 228., 129., 325.]], dtype=float32),
        'labels': array([11,  0], dtype=int64), 'bboxes_ignore': array([], shape=(0, 4), dtype=float32),
        'masks': [[], []], 'seg_map': '000921.png'},
        以下六个key是在self.pre_pipeline中新增,并根据配置文件中的值初始化的
        'img_prefix': 'd:/mmdetection/data/yexi/images/',
        'seg_prefix': None,
        'proposal_file': None,
        'bbox_fields': [],
        'mask_fields': [],
        'seg_fields': []}
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        """
        results = {
        'img_info': {'file_name': '000921.jpg', 'id': 754, 'width': 800, 'height': 600, 'filename': '000921.jpg'},
        'ann_info': {'bboxes': array([[108., 350., 149., 455.],[ 68., 228., 129., 325.]], dtype=float32), 
        'labels': array([11,  0], dtype=int64), 'bboxes_ignore': array([], shape=(0, 4), dtype=float32), 
        'masks': [[], []], 'seg_map': '000921.png'}}
        """
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    @classmethod
    def get_classes(cls, classes=None):
        """获取当前数据集的类名.

        Args:
            classes (Sequence[str] | str | None): 如果 classes 为 None,则使用内置数据集定义的默认 CLASSES.
                如果 classes 是字符串,则将其作为文件名.该文件包含类的名称,其中每一行包含一个类名.
                如果 classes 是元组或列表,那么覆盖数据集定义的 CLASSES.

        Returns:
            tuple[str] or list[str]: 数据集的类别名称.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # 将其作为文件路径并加载其内容
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def get_cat2imgs(self):
        """Get a dict with class as key and img_ids as values, which will be
        used in :class:`ClassAwareSampler`.

        Returns:
            dict[list]: A dict of per-label image list,
            the item of the dict indicates a label index,
            corresponds to the image index that contains the label.
        """
        if self.CLASSES is None:
            raise ValueError('self.CLASSES can not be None')
        # sort the label index
        cat2imgs = {i: [] for i in range(len(self.CLASSES))}
        for i in range(len(self)):
            cat_ids = set(self.get_cat_ids(i))
            for cat in cat_ids:
                cat2imgs[cat].append(i)
        return cat2imgs

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (f'\n{self.__class__.__name__} {dataset_type} dataset '
                  f'with number of images {len(self)}, '
                  f'and instance counts: \n')
        if self.CLASSES is None:
            result += 'Category names are not provided. \n'
            return result
        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
        # count the instance number in each image
        for idx in range(len(self)):
            label = self.get_ann_info(idx)['labels']
            unique, counts = np.unique(label, return_counts=True)
            if len(unique) > 0:
                # add the occurrence number to each class
                instance_count[unique] += counts
            else:
                # background is the last index
                instance_count[-1] += 1
        # create a table with category count
        table_data = [['category', 'count'] * 5]
        row_data = []
        for cls, count in enumerate(instance_count):
            if cls < len(self.CLASSES):
                row_data += [f'{cls} [{self.CLASSES[cls]}]', f'{count}']
            else:
                # add the background number
                row_data += ['-1 background', f'{count}']
            if len(row_data) == 10:
                table_data.append(row_data)
                row_data = []
        if len(row_data) >= 2:
            if row_data[-1] == '0':
                row_data = row_data[:-2]
            if len(row_data) >= 2:
                table_data.append([])
                table_data.append(row_data)

        table = AsciiTable(table_data)
        result += table.table
        return result
