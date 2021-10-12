import itertools
import logging
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from huicv.evaluation.expand_cocofmt_eval import COCOExpandEval
from huicv.evaluation.location_evaluation import LocationEvaluator
from functools import partial

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .coco import CocoDataset


# add by hui, if there is not corner dataset, create one
def generate_corner_json_file_if_not_exist(ann_file, data_root, corner_kwargs):
    from huicv.corner_dataset.corner_dataset_util import generate_corner_dataset

    # generate corner json file name
    if data_root is not None:
        if not osp.isabs(ann_file):
            ann_file = osp.join(data_root, ann_file)
    origin_ann_file = ann_file
    max_tile_size, tile_overlap = corner_kwargs['max_tile_size'], corner_kwargs['tile_overlap']
    ann_file = "{}_corner_w{}h{}ow{}oh{}.json".format(
        ann_file[:-5], max_tile_size[0], max_tile_size[1], tile_overlap[0], tile_overlap[1])
    ann_dir, ann_file_name = osp.split(ann_file)
    corner_file_dir = osp.join(ann_dir, 'corner')
    ann_file = osp.join(corner_file_dir, ann_file_name)

    # generate corner dataset and save to disk, if it not exists
    if not osp.exists(ann_file):
        _ = generate_corner_dataset(origin_ann_file, save_path=ann_file, **corner_kwargs)
        print("generate corner dataset done, please re-run your code.")
        exit(0)
    return ann_file


def generate_pesudo_bbox_for_noise_data(ann_file, data_root, noise_kwargs):
    from huicv.coarse_utils.noise_data_utils import get_new_json_file_path, generate_pseudo_bbox_for_point
    # ann_file, _ = get_new_json_file_path(ann_file, data_root, 'noise', 'noisept')
    # assert osp.exists(ann_file), "{} not exist.".format(ann_file)
    ori_ann_file = ann_file
    pseudo_wh = noise_kwargs['pseudo_wh']
    if isinstance(pseudo_wh, (int, float)):
        noise_kwargs['pseudo_wh'] = pseudo_wh = (pseudo_wh, pseudo_wh)
    suffix = 'pseuw{}h{}'.format(*pseudo_wh)
    ann_file, _ = get_new_json_file_path(ori_ann_file, data_root, None, suffix)
    if not osp.exists(ann_file):
        _ = generate_pseudo_bbox_for_point(ori_ann_file, ann_file, **noise_kwargs)
        print("generate pseudo bbox for dataset done, please re-run your code.")
        exit(0)
    return ann_file


@DATASETS.register_module()
class CocoFmtDataset(CocoDataset):

    CLASSES = None

    def __init__(self,
                 ann_file,
                 data_root=None,
                 corner_kwargs=None,
                 train_ignore_as_bg=True,
                 noise_kwargs=None,
                 merge_after_infer_kwargs=None,
                 **kwargs):
        # add by hui, if there is not corner dataset, create one
        if corner_kwargs is not None:
            assert ann_file[-5:] == '.json', "ann_file must be a json file."
            ann_file = generate_corner_json_file_if_not_exist(ann_file, data_root, corner_kwargs)
            print("load corner dataset json file from {}".format(ann_file))
        if noise_kwargs is not None:
            if 'pseudo_wh' in noise_kwargs and noise_kwargs['pseudo_wh'] is not None:
                ann_file = generate_pesudo_bbox_for_noise_data(ann_file, data_root, noise_kwargs)
            elif 'wh_suffix' in noise_kwargs:
                from .noise_data_utils import get_new_json_file_path
                ann_file, _ = get_new_json_file_path(ann_file, data_root, noise_kwargs['sub_dir'],
                                                     noise_kwargs['wh_suffix'])
            else:
                raise ValueError('one of [pseudo_wh, wh_suffix] must be given')
            print("load noise dataset json file from {}".format(ann_file))

        self.train_ignore_as_bg = train_ignore_as_bg
        self.merge_after_infer_kwargs = merge_after_infer_kwargs

        super(CocoFmtDataset, self).__init__(
            ann_file,
            data_root=data_root,
            **kwargs
        )

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        if self.CLASSES is None:
            self.CLASSES = [cat['name'] for cat in self.coco.dataset['categories']]  # add by hui
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def _filter_imgs(self, min_size=32):
        valid_inds = super(CocoFmtDataset, self)._filter_imgs(min_size)
        print("valid image count: ", len(valid_inds))   # add by hui
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if self.train_ignore_as_bg and ann.get('ignore', False):  # change by hui
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)
        return ann

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 cocofmt_kwargs={}, use_location_metric=False, location_kwargs={}):  # add by hui
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                # add by hui ######################################################
                merge_after_infer_kwargs = self.merge_after_infer_kwargs
                if merge_after_infer_kwargs is not None:  # merge result before eval
                    from huicv.evaluation.evaluate_tiny import merge_det_result
                    merge_gt_file = merge_after_infer_kwargs.get("merge_gt_file")
                    merge_nms_th = merge_after_infer_kwargs.get("merge_nms_th", 0.5)
                    cocoGt, result_files[metric] = merge_det_result(result_files[metric], self.ann_file, merge_gt_file,
                                                                    merge_nms_th)
                ###################################################################
                predictions = mmcv.load(result_files[metric])
                # add by hui ######################################################
                import shutil
                shutil.copy(result_files[metric], './exp/latest_result.json')
                ######################################################
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            # add by hui for location evaluation ####################################
            if use_location_metric:
                location_eval = LocationEvaluator(**location_kwargs)
                print(location_eval.__dict__)
                LocationEvaluator.add_center_from_bbox_if_no_point(cocoDt)
                res_set = location_eval(cocoDt, cocoGt)
                location_eval.summarize(res_set, cocoGt, print_func=partial(print_log, logger=logger))
                continue
            # ########################################################################

            iou_type = 'bbox' if metric == 'proposal' else metric
            # change by hui ##################################################
            # cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval = COCOExpandEval(cocoGt, cocoDt, iou_type, **cocofmt_kwargs)
            param_kwargs = cocofmt_kwargs['cocofmt_param'] if 'cocofmt_param' in cocofmt_kwargs else {}
            if 'cat_ids' not in param_kwargs: cocoEval.params.catIds = self.cat_ids
            if 'img_ids' not in param_kwargs: cocoEval.params.imgIds = self.img_ids
            if 'maxDets' not in param_kwargs: cocoEval.params.maxDets = list(proposal_nums)
            if 'iouThrs' not in param_kwargs: cocoEval.params.iouThrs = np.array(iou_thrs)
            print(cocoEval.__dict__)
            print({k:v for k, v in cocoEval.params.__dict__.items() if k not in ['imgIds']})
            ###################################################################

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                cocoEval.summarize(print_func=partial(print_log, logger=logger))   # add by hui
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
