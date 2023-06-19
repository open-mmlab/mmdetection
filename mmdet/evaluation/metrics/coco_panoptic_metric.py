# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from typing import Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCOPanoptic
from mmdet.registry import METRICS
from ..functional import (INSTANCE_OFFSET, pq_compute_multi_core,
                          pq_compute_single_core)

try:
    import panopticapi
    from panopticapi.evaluation import VOID, PQStat
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    panopticapi = None
    id2rgb = None
    rgb2id = None
    VOID = None
    PQStat = None


@METRICS.register_module()
class CocoPanopticMetric(BaseMetric):
    """COCO panoptic segmentation evaluation metric.

    Evaluate PQ, SQ RQ for panoptic segmentation tasks. Please refer to
    https://cocodataset.org/#panoptic-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        seg_prefix (str, optional): Path to the directory which contains the
            coco panoptic segmentation mask. It should be specified when
            evaluate. Defaults to None.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created.
            It should be specified when format_only is True. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        nproc (int): Number of processes for panoptic quality computing.
            Defaults to 32. When ``nproc`` exceeds the number of cpu cores,
            the number of cpu cores is used.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = 'coco_panoptic'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 seg_prefix: Optional[str] = None,
                 classwise: bool = False,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 nproc: int = 32,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        if panopticapi is None:
            raise RuntimeError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')

        super().__init__(collect_device=collect_device, prefix=prefix)
        self.classwise = classwise
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.tmp_dir = None
        # outfile_prefix should be a prefix of a path which points to a shared
        # storage when train or test with multi nodes.
        self.outfile_prefix = outfile_prefix
        if outfile_prefix is None:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.outfile_prefix = osp.join(self.tmp_dir.name, 'results')
        # the directory to save predicted panoptic segmentation mask
        self.seg_out_dir = f'{self.outfile_prefix}.panoptic'
        self.nproc = nproc
        self.seg_prefix = seg_prefix

        self.cat_ids = None
        self.cat2label = None

        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

        if ann_file:
            with get_local_path(
                    ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = COCOPanoptic(local_path)
            self.categories = self._coco_api.cats
        else:
            self._coco_api = None
            self.categories = None

    def __del__(self) -> None:
        """Clean up."""
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> Tuple[str, str]:
        """Convert ground truth to coco panoptic segmentation format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset.
            outfile_prefix (str): The filename prefix of the json file. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".

        Returns:
            Tuple[str, str]: The filename of the json file and the name of the\
                directory which contains panoptic segmentation masks.
        """
        assert len(gt_dicts) > 0, 'gt_dicts is empty.'
        gt_folder = osp.dirname(gt_dicts[0]['seg_map_path'])
        converted_json_path = f'{outfile_prefix}.gt.json'

        categories = []
        for id, name in enumerate(self.dataset_meta['classes']):
            isthing = 1 if name in self.dataset_meta['thing_classes'] else 0
            categories.append({'id': id, 'name': name, 'isthing': isthing})

        image_infos = []
        annotations = []
        for gt_dict in gt_dicts:
            img_id = gt_dict['image_id']
            image_info = {
                'id': img_id,
                'width': gt_dict['width'],
                'height': gt_dict['height'],
                'file_name': osp.split(gt_dict['seg_map_path'])[-1]
            }
            image_infos.append(image_info)

            pan_png = mmcv.imread(gt_dict['seg_map_path']).squeeze()
            pan_png = pan_png[:, :, ::-1]
            pan_png = rgb2id(pan_png)
            segments_info = []
            for segment_info in gt_dict['segments_info']:
                id = segment_info['id']
                label = segment_info['category']
                mask = pan_png == id
                isthing = categories[label]['isthing']
                if isthing:
                    iscrowd = 1 if not segment_info['is_thing'] else 0
                else:
                    iscrowd = 0

                new_segment_info = {
                    'id': id,
                    'category_id': label,
                    'isthing': isthing,
                    'iscrowd': iscrowd,
                    'area': mask.sum()
                }
                segments_info.append(new_segment_info)

            segm_file = image_info['file_name'].replace('jpg', 'png')
            annotation = dict(
                image_id=img_id,
                segments_info=segments_info,
                file_name=segm_file)
            annotations.append(annotation)
            pan_png = id2rgb(pan_png)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoPanopticMetric.'
        )
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        dump(coco_json, converted_json_path)
        return converted_json_path, gt_folder

    def result2json(self, results: Sequence[dict],
                    outfile_prefix: str) -> Tuple[str, str]:
        """Dump the panoptic results to a COCO style json file and a directory.

        Args:
            results (Sequence[dict]): Testing results of the dataset.
            outfile_prefix (str): The filename prefix of the json files and the
                directory.

        Returns:
            Tuple[str, str]: The json file and the directory which contains \
                panoptic segmentation masks. The filename of the json is
                "somepath/xxx.panoptic.json" and name of the directory is
                "somepath/xxx.panoptic".
        """
        label2cat = dict((v, k) for (k, v) in self.cat2label.items())
        pred_annotations = []
        for idx in range(len(results)):
            result = results[idx]
            for segment_info in result['segments_info']:
                sem_label = segment_info['category_id']
                # convert sem_label to json label
                cat_id = label2cat[sem_label]
                segment_info['category_id'] = label2cat[sem_label]
                is_thing = self.categories[cat_id]['isthing']
                segment_info['isthing'] = is_thing
            pred_annotations.append(result)
        pan_json_results = dict(annotations=pred_annotations)
        json_filename = f'{outfile_prefix}.panoptic.json'
        dump(pan_json_results, json_filename)
        return json_filename, (
            self.seg_out_dir
            if self.tmp_dir is None else tempfile.gettempdir())

    def _parse_predictions(self,
                           pred: dict,
                           img_id: int,
                           segm_file: str,
                           label2cat=None) -> dict:
        """Parse panoptic segmentation predictions.

        Args:
            pred (dict): Panoptic segmentation predictions.
            img_id (int): Image id.
            segm_file (str): Segmentation file name.
            label2cat (dict): Mapping from label to category id.
                Defaults to None.

        Returns:
            dict: Parsed predictions.
        """
        result = dict()
        result['img_id'] = img_id
        # shape (1, H, W) -> (H, W)
        pan = pred['pred_panoptic_seg']['sem_seg'].cpu().numpy()[0]
        ignore_index = pred['pred_panoptic_seg'].get(
            'ignore_index', len(self.dataset_meta['classes']))
        pan_labels = np.unique(pan)
        segments_info = []
        for pan_label in pan_labels:
            sem_label = pan_label % INSTANCE_OFFSET
            # We reserve the length of dataset_meta['classes']
            # and ignore_index for VOID label
            if sem_label == len(
                    self.dataset_meta['classes']) or sem_label == ignore_index:
                continue
            mask = pan == pan_label
            area = mask.sum()
            segments_info.append({
                'id':
                int(pan_label),
                # when ann_file provided, sem_label should be cat_id, otherwise
                # sem_label should be a continuous id, not the cat_id
                # defined in dataset
                'category_id':
                label2cat[sem_label] if label2cat else sem_label,
                'area':
                int(area)
            })
        # evaluation script uses 0 for VOID label.
        pan[pan % INSTANCE_OFFSET == len(self.dataset_meta['classes'])] = VOID
        pan[pan % INSTANCE_OFFSET == ignore_index] = VOID

        pan = id2rgb(pan).astype(np.uint8)
        mmcv.imwrite(pan[:, :, ::-1], osp.join(self.seg_out_dir, segm_file))
        result = {
            'image_id': img_id,
            'segments_info': segments_info,
            'file_name': segm_file
        }

        return result

    def _compute_batch_pq_stats(self, data_samples: Sequence[dict]):
        """Process gts and predictions when ``outfile_prefix`` is not set, gts
        are from dataset or a json file which is defined by ``ann_file``.

        Intermediate results, ``pq_stats``, are computed here and put into
        ``self.results``.
        """
        if self._coco_api is None:
            categories = dict()
            for id, name in enumerate(self.dataset_meta['classes']):
                isthing = 1 if name in self.dataset_meta['thing_classes']\
                    else 0
                categories[id] = {'id': id, 'name': name, 'isthing': isthing}
            label2cat = None
        else:
            categories = self.categories
            cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
            label2cat = {i: cat_id for i, cat_id in enumerate(cat_ids)}

        for data_sample in data_samples:
            # parse pred
            img_id = data_sample['img_id']
            segm_file = osp.basename(data_sample['img_path']).replace(
                'jpg', 'png')
            result = self._parse_predictions(
                pred=data_sample,
                img_id=img_id,
                segm_file=segm_file,
                label2cat=label2cat)

            # parse gt
            gt = dict()
            gt['image_id'] = img_id
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['file_name'] = segm_file

            if self._coco_api is None:
                # get segments_info from data_sample
                seg_map_path = osp.join(self.seg_prefix, segm_file)
                pan_png = mmcv.imread(seg_map_path).squeeze()
                pan_png = pan_png[:, :, ::-1]
                pan_png = rgb2id(pan_png)
                segments_info = []

                for segment_info in data_sample['segments_info']:
                    id = segment_info['id']
                    label = segment_info['category']
                    mask = pan_png == id
                    isthing = categories[label]['isthing']
                    if isthing:
                        iscrowd = 1 if not segment_info['is_thing'] else 0
                    else:
                        iscrowd = 0

                    new_segment_info = {
                        'id': id,
                        'category_id': label,
                        'isthing': isthing,
                        'iscrowd': iscrowd,
                        'area': mask.sum()
                    }
                    segments_info.append(new_segment_info)
            else:
                # get segments_info from annotation file
                segments_info = self._coco_api.imgToAnns[img_id]

            gt['segments_info'] = segments_info

            pq_stats = pq_compute_single_core(
                proc_id=0,
                annotation_set=[(gt, result)],
                gt_folder=self.seg_prefix,
                pred_folder=self.seg_out_dir,
                categories=categories,
                backend_args=self.backend_args)

            self.results.append(pq_stats)

    def _process_gt_and_predictions(self, data_samples: Sequence[dict]):
        """Process gts and predictions when ``outfile_prefix`` is set.

        The predictions will be saved to directory specified by
        ``outfile_predfix``. The matched pair (gt, result) will be put into
        ``self.results``.
        """
        for data_sample in data_samples:
            # parse pred
            img_id = data_sample['img_id']
            segm_file = osp.basename(data_sample['img_path']).replace(
                'jpg', 'png')
            result = self._parse_predictions(
                pred=data_sample, img_id=img_id, segm_file=segm_file)

            # parse gt
            gt = dict()
            gt['image_id'] = img_id
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]

            if self._coco_api is None:
                # get segments_info from dataset
                gt['segments_info'] = data_sample['segments_info']
                gt['seg_map_path'] = data_sample['seg_map_path']

            self.results.append((gt, result))

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        # If ``self.tmp_dir`` is none, it will save gt and predictions to
        # self.results, otherwise, it will compute pq_stats here.
        if self.tmp_dir is None:
            self._process_gt_and_predictions(data_samples)
        else:
            self._compute_batch_pq_stats(data_samples)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch. There
                are two cases:

                - When ``outfile_prefix`` is not provided, the elements in
                  results are pq_stats which can be summed directly to get PQ.
                - When ``outfile_prefix`` is provided, the elements in
                  results are tuples like (gt, pred).

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.tmp_dir is None:
            # do evaluation after collect all the results

            # split gt and prediction list
            gts, preds = zip(*results)

            if self._coco_api is None:
                # use converted gt json file to initialize coco api
                logger.info('Converting ground truth to coco format...')
                coco_json_path, gt_folder = self.gt_to_coco_json(
                    gt_dicts=gts, outfile_prefix=self.outfile_prefix)
                self._coco_api = COCOPanoptic(coco_json_path)
            else:
                gt_folder = self.seg_prefix

            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
            self.cat2label = {
                cat_id: i
                for i, cat_id in enumerate(self.cat_ids)
            }
            self.img_ids = self._coco_api.get_img_ids()
            self.categories = self._coco_api.cats

            # convert predictions to coco format and dump to json file
            json_filename, pred_folder = self.result2json(
                results=preds, outfile_prefix=self.outfile_prefix)

            if self.format_only:
                logger.info('results are saved in '
                            f'{osp.dirname(self.outfile_prefix)}')
                return dict()

            imgs = self._coco_api.imgs
            gt_json = self._coco_api.img_ann_map
            gt_json = [{
                'image_id': k,
                'segments_info': v,
                'file_name': imgs[k]['segm_file']
            } for k, v in gt_json.items()]
            pred_json = load(json_filename)
            pred_json = dict(
                (el['image_id'], el) for el in pred_json['annotations'])

            # match the gt_anns and pred_anns in the same image
            matched_annotations_list = []
            for gt_ann in gt_json:
                img_id = gt_ann['image_id']
                if img_id not in pred_json.keys():
                    raise Exception('no prediction for the image'
                                    ' with id: {}'.format(img_id))
                matched_annotations_list.append((gt_ann, pred_json[img_id]))

            pq_stat = pq_compute_multi_core(
                matched_annotations_list,
                gt_folder,
                pred_folder,
                self.categories,
                backend_args=self.backend_args,
                nproc=self.nproc)

        else:
            # aggregate the results generated in process
            if self._coco_api is None:
                categories = dict()
                for id, name in enumerate(self.dataset_meta['classes']):
                    isthing = 1 if name in self.dataset_meta[
                        'thing_classes'] else 0
                    categories[id] = {
                        'id': id,
                        'name': name,
                        'isthing': isthing
                    }
                self.categories = categories

            pq_stat = PQStat()
            for result in results:
                pq_stat += result

        metrics = [('All', None), ('Things', True), ('Stuff', False)]
        pq_results = {}

        for name, isthing in metrics:
            pq_results[name], classwise_results = pq_stat.pq_average(
                self.categories, isthing=isthing)
            if name == 'All':
                pq_results['classwise'] = classwise_results

        classwise_results = None
        if self.classwise:
            classwise_results = {
                k: v
                for k, v in zip(self.dataset_meta['classes'],
                                pq_results['classwise'].values())
            }

        print_panoptic_table(pq_results, classwise_results, logger=logger)
        results = parse_pq_results(pq_results)

        return results


def parse_pq_results(pq_results: dict) -> dict:
    """Parse the Panoptic Quality results.

    Args:
        pq_results (dict): Panoptic Quality results.

    Returns:
        dict: Panoptic Quality results parsed.
    """
    result = dict()
    result['PQ'] = 100 * pq_results['All']['pq']
    result['SQ'] = 100 * pq_results['All']['sq']
    result['RQ'] = 100 * pq_results['All']['rq']
    result['PQ_th'] = 100 * pq_results['Things']['pq']
    result['SQ_th'] = 100 * pq_results['Things']['sq']
    result['RQ_th'] = 100 * pq_results['Things']['rq']
    result['PQ_st'] = 100 * pq_results['Stuff']['pq']
    result['SQ_st'] = 100 * pq_results['Stuff']['sq']
    result['RQ_st'] = 100 * pq_results['Stuff']['rq']
    return result


def print_panoptic_table(
        pq_results: dict,
        classwise_results: Optional[dict] = None,
        logger: Optional[Union['MMLogger', str]] = None) -> None:
    """Print the panoptic evaluation results table.

    Args:
        pq_results(dict): The Panoptic Quality results.
        classwise_results(dict, optional): The classwise Panoptic Quality.
            results. The keys are class names and the values are metrics.
            Defaults to None.
        logger (:obj:`MMLogger` | str, optional): Logger used for printing
            related information during evaluation. Default: None.
    """

    headers = ['', 'PQ', 'SQ', 'RQ', 'categories']
    data = [headers]
    for name in ['All', 'Things', 'Stuff']:
        numbers = [
            f'{(pq_results[name][k] * 100):0.3f}' for k in ['pq', 'sq', 'rq']
        ]
        row = [name] + numbers + [pq_results[name]['n']]
        data.append(row)
    table = AsciiTable(data)
    print_log('Panoptic Evaluation Results:\n' + table.table, logger=logger)

    if classwise_results is not None:
        class_metrics = [(name, ) + tuple(f'{(metrics[k] * 100):0.3f}'
                                          for k in ['pq', 'sq', 'rq'])
                         for name, metrics in classwise_results.items()]
        num_columns = min(8, len(class_metrics) * 4)
        results_flatten = list(itertools.chain(*class_metrics))
        headers = ['category', 'PQ', 'SQ', 'RQ'] * (num_columns // 4)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::num_columns] for i in range(num_columns)])
        data = [headers]
        data += [result for result in results_2d]
        table = AsciiTable(data)
        print_log(
            'Classwise Panoptic Evaluation Results:\n' + table.table,
            logger=logger)
