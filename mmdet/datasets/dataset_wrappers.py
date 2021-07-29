import bisect
import copy
import math
import random
from collections import defaultdict

import mmcv
import numpy as np
from mmcv.utils import print_log
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS
from .coco import CocoDataset
from .pipelines import Compose


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.separate_eval = separate_eval
        if not separate_eval:
            if any([isinstance(ds, CocoDataset) for ds in datasets]):
                raise NotImplementedError(
                    'Evaluating concatenated CocoDataset as a whole is not'
                    ' supported! Please set "separate_eval=True"')
            elif len(set([type(ds) for ds in datasets])) != 1:
                raise NotImplementedError(
                    'All the datasets should have same types')

        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)

    def get_cat_ids(self, idx):
        """Get category ids of concatenated dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_cat_ids(sample_idx)

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the results.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: AP results of the total dataset or each separate
            dataset if `self.separate_eval=True`.
        """
        assert len(results) == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                f'{type(dataset)} does not implement evaluate function'

        if self.separate_eval:
            dataset_idx = -1
            total_eval_results = dict()
            for size, dataset in zip(self.cumulative_sizes, self.datasets):
                start_idx = 0 if dataset_idx == -1 else \
                    self.cumulative_sizes[dataset_idx]
                end_idx = self.cumulative_sizes[dataset_idx + 1]

                results_per_dataset = results[start_idx:end_idx]
                print_log(
                    f'\nEvaluateing {dataset.ann_file} with '
                    f'{len(results_per_dataset)} images now',
                    logger=logger)

                eval_results_per_dataset = dataset.evaluate(
                    results_per_dataset, logger=logger, **kwargs)
                dataset_idx += 1
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f'{dataset_idx}_{k}': v})

            return total_eval_results
        elif any([isinstance(ds, CocoDataset) for ds in self.datasets]):
            raise NotImplementedError(
                'Evaluating concatenated CocoDataset as a whole is not'
                ' supported! Please set "separate_eval=True"')
        elif len(set([type(ds) for ds in self.datasets])) != 1:
            raise NotImplementedError(
                'All the datasets should have same types')
        else:
            original_data_infos = self.datasets[0].data_infos
            self.datasets[0].data_infos = sum(
                [dataset.data_infos for dataset in self.datasets], [])
            eval_results = self.datasets[0].evaluate(
                results, logger=logger, **kwargs)
            self.datasets[0].data_infos = original_data_infos
            return eval_results


@DATASETS.register_module()
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def get_cat_ids(self, idx):
        """Get category ids of repeat dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.dataset.get_cat_ids(idx % self._ori_len)

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len


# Modified from https://github.com/facebookresearch/detectron2/blob/41d475b75a230221e21d9cac5d69655e3415e3a4/detectron2/data/samplers/distributed_sampler.py#L57 # noqa
@DATASETS.register_module()
class ClassBalancedDataset:
    """A wrapper of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :func:`self.get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes will not be oversampled. Otherwise, they will be categorized
            as the pure background class and involved into the oversampling.
            Default: True.
    """

    def __init__(self, dataset, oversample_thr, filter_empty_gt=True):
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = dataset.CLASSES

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, repeat_thr):
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (:obj:`CustomDataset`): The dataset
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """

        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = defaultdict(int)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            repeat_factor = 1
            if len(cat_ids) > 0:
                repeat_factor = max(
                    {category_repeat[cat_id]
                     for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        """Length after repetition."""
        return len(self.repeat_indices)


def filter_box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return ((w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
            & (ar < ar_thr))


@DATASETS.register_module()
class MosaicMixUpDataset:
    """MosaicMixUp dataset used in YOLOX

                        mosaic transform
                          center_x
                ______________________________
               \          pad     \  pad     \
               \       ___________\          \
               \      \           \          \
               \      \  image1   \________  \
               \      \           \        \ \
               \      \           \ image2 \ \
   center_y    \------------------\----------\
               \    \  cropped    \          \
               \pad \   image3    \  image4  \
               \____\________________________\
                    \             \
                    \_____________\
    Args:
        dataset (CunstomDataset): Child of CunstomDataset which can get image and annotations by index.
        mosaic_pipeline (pipeline): Augmentations after mosaic
        pipeline (pipeline): Augmentations after mosaic and mixup.
        img_scale (Sequence[int]): image size after mosaic pipeline
        enable_mosaic (bool): enable/disable mosiac aug, controlled by yolox process hook
        mosaic_scale (Sequence[float]): center range of mosaic output
        enable_mixup (bool): enable/disable mixup aug, controlled by yolox process hook
        mixup_scale (Sequence[float]): image scale factor of mixup
        pad_value (int): pad value

    """
    def __init__(self,
                 dataset,
                 mosaic_pipeline=None,
                 mixup_pipeline=None,
                 pipeline=None,
                 img_scale=(640, 640),
                 enable_mosaic=True,
                 mosaic_scale=(0.5, 1.5),
                 enable_mixup=True,
                 mixup_scale=(0.5, 1.5),
                 pad_value=114):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = dataset.flag
        self.mosaic_pipeline = Compose(
            mosaic_pipeline) if mosaic_pipeline is not None else None
        self.mixup_pipeline = Compose(
            mixup_pipeline) if mixup_pipeline is not None else None
        self.pipeline = Compose(pipeline) if pipeline is not None else None
        self.enable_mosaic = enable_mosaic
        self.mosaic_scale = mosaic_scale
        self.enable_mixup = enable_mixup
        self.mixup_scale = mixup_scale
        self.mixup_flip_ratio = 0.5
        self.pad_value = pad_value
        self.img_scale = img_scale  # h,w
        self.dynamic_scale = img_scale
        self.num_sample = len(dataset)

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        results = self.dataset[idx]
        if self.enable_mosaic:
            results = self.mosiac(results)
            results = self.mosaic_pipeline(results)
        if self.enable_mixup and results['gt_bboxes'].shape[0] > 0:
            results = self.mixup(results)
        # dynamic resize
        results['scale'] = self.dynamic_scale
        results = self.pipeline(results)
        return results

    def mosiac(self, results):
        mosaic_labels = []
        mosaic_bboxes = []
        mosaic_img = np.full((self.img_scale[0] * 2, self.img_scale[1] * 2, 3),
                             self.pad_value,
                             dtype=np.uint8)

        # mosaic center x, y
        center_x = int(random.uniform(*self.mosaic_scale) * self.img_scale[0])
        center_y = int(random.uniform(*self.mosaic_scale) * self.img_scale[1])
        center_position = (center_x, center_y)

        indices = [random.randint(0, self.num_sample - 1) for _ in range(3)]

        for i, loc in enumerate(
            ('top_left', 'top_right', 'bottom_left', 'bottom_right')):
            if loc == 'top_left':
                results = copy.deepcopy(results)
            else:
                results = copy.deepcopy(self.dataset[indices[i - 1]])

            img_i = results['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosiac_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results['gt_bboxes']
            gt_labels_i = results['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:,
                            0::2] = scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:,
                            1::2] = scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0,
                                             2 * self.img_scale[1])
            mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0,
                                             2 * self.img_scale[0])
            mosaic_labels = np.concatenate(mosaic_labels, 0)

        results['img'] = mosaic_img.astype(np.float32)
        results['img_shape'] = mosaic_img.shape
        results['ori_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels

        return results

    def _mosiac_combine(self, loc, center_position_xy, img_shape_wh):
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), max(
                center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], max(center_position_xy[1] - img_shape_wh[1], 0), min(
                center_position_xy[0] + img_shape_wh[0],
                self.img_scale[1] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(
                center_position_xy[0] - img_shape_wh[0],
                0), center_position_xy[1], center_position_xy[0], min(
                    self.img_scale[0] * 2,
                    center_position_xy[1] + img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], center_position_xy[1], min(
                center_position_xy[0] + img_shape_wh[0],
                self.img_scale[0] * 2), min(
                    self.img_scale[1] * 2,
                    center_position_xy[1] + img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def mixup(self, results):
        results = copy.deepcopy(results)

        jit_factor = random.uniform(*self.mixup_scale)
        is_filp = random.uniform(0, 1) > self.mixup_flip_ratio

        # 0. retrieve data
        gt_bboxes_i = []
        for i in range(15):
            index = random.randint(0, self.num_sample - 1)
            gt_bboxes_i = self.dataset.get_ann_info(index)['bboxes']
            if len(gt_bboxes_i) != 0:
                break

        if len(gt_bboxes_i) == 0:
            return results

        retrieve_results = copy.deepcopy(self.dataset[index])
        retrieve_img = retrieve_results['img']

        if len(retrieve_img.shape) == 3:
            out_img = np.ones((self.dynamic_scale[0], self.dynamic_scale[1],
                               3)) * self.pad_value
        else:
            out_img = np.ones(self.dynamic_scale) * self.pad_value

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0],
                          self.dynamic_scale[1] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        origin_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w,
                                          target_w), 3)).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes[:, 0::2] = np.clip(
            retrieve_gt_bboxes[:, 0::2] * scale_ratio, 0, origin_w)
        retrieve_gt_bboxes[:, 1::2] = np.clip(
            retrieve_gt_bboxes[:, 1::2] * scale_ratio, 0, origin_h)

        if is_filp:
            retrieve_gt_bboxes[:, 0::2] = (
                origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1])

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        cp_retrieve_gt_bboxes[:, 0::2] = np.clip(
            cp_retrieve_gt_bboxes[:, 0::2] - x_offset, 0, target_w)
        cp_retrieve_gt_bboxes[:, 1::2] = np.clip(
            cp_retrieve_gt_bboxes[:, 1::2] - y_offset, 0, target_h)
        keep_list = filter_box_candidates(cp_retrieve_gt_bboxes.T,
                                          retrieve_gt_bboxes.T, 5)

        # 8. mix up
        if keep_list.sum() >= 1.0:
            origin_img = origin_img.astype(np.float32)
            mixup_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(
                np.float32)

            retrieve_gt_labels = retrieve_results['gt_labels'][keep_list]
            retrieve_gt_bboxes = cp_retrieve_gt_bboxes[keep_list]
            mixup_gt_bboxes = np.concatenate(
                (results['gt_bboxes'], retrieve_gt_bboxes), axis=0)
            mixup_gt_labels = np.concatenate(
                (results['gt_labels'], retrieve_gt_labels), axis=0)

            results['img'] = mixup_img
            results['gt_bboxes'] = mixup_gt_bboxes
            results['gt_labels'] = mixup_gt_labels

        return results
