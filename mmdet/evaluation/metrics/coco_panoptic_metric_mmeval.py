# Copyright (c) OpenMMLab. All rights reserved.

import sys
sys.path.append('/home/PJLAB/liangyiwen/Even/code/mmeval_exp/mmeval')
from mmeval.metrics.coco_panoptic import COCOPanopticMetric
import datetime
import itertools
import os.path as osp
import tempfile
from typing import Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmengine.fileio import FileClient, dump, load
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable
from collections import OrderedDict, defaultdict

from mmdet.datasets.api_wrappers import COCOPanoptic
from mmdet.registry import METRICS
from ..functional import INSTANCE_OFFSET

try:
    from panopticapi.evaluation import VOID
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    panopticapi = None
    id2rgb = None
    rgb2id = None
    VOID = None
    PQStat = None



@METRICS.register_module()
class CocoPanopticMetricMMEval(COCOPanopticMetric):
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
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='local')``.
    """
    default_prefix: Optional[str] = 'coco_panoptic'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 seg_prefix: Optional[str] = None,
                 classwise: bool = False,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 nproc: int = 32,
                 file_client_args: dict = dict(backend='local')):

        super().__init__(ann_file=ann_file, outfile_prefix=outfile_prefix, gt_folder=seg_prefix,
                         classwise=classwise, nproc=nproc, backend_args=file_client_args)

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
        self.pred_folder = self.seg_out_dir
        self.seg_prefix = seg_prefix

        self.cat_ids = None
        self.cat2label = None

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
        pan_labels = np.unique(pan)
        segments_info = []
        for pan_label in pan_labels:
            sem_label = pan_label % INSTANCE_OFFSET
            mask = pan == pan_label
            area = mask.sum()
            segments_info.append({
                'id':
                int(pan_label // INSTANCE_OFFSET),
                # when ann_file provided, sem_label should be cat_id, otherwise
                # sem_label should be a continuous id, not the cat_id
                # defined in dataset
                'category_id':
                label2cat[sem_label] if label2cat else sem_label,
                'area':
                int(area)
            })
        # evaluation script uses 0 for VOID label.
        pan[pan % INSTANCE_OFFSET > len(self.dataset_meta['classes'])] = VOID
        pan = id2rgb(pan // INSTANCE_OFFSET).astype(np.uint8)
        mmcv.imwrite(pan[:, :, ::-1], osp.join(self.seg_out_dir, segm_file))
        result = {
            'image_id': img_id,
            'segments_info': segments_info,
            'file_name': segm_file
        }
        return result

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The matched pair (gt, result) 
        should be stored in ``self._results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        predictions, groundtruths = [], []
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

            if self.ann_file is None:
                # get segments_info from dataset
                gt['segments_info'] = data_sample['segments_info']
                gt['seg_map_path'] = data_sample['seg_map_path']

            predictions.append(result)
            groundtruths.append(gt)

        self.add(predictions, groundtruths)

    def results2json(self, results: Sequence[dict],
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

        _coco_api = COCOPanoptic(self.ann_file)
        cat_ids = _coco_api.get_cat_ids(
            cat_names=self.dataset_meta['classes'])
        cat2label = {
            cat_id: i
            for i, cat_id in enumerate(cat_ids)
        }
        label2cat = dict((v, k) for (k, v) in cat2label.items())
        categories = _coco_api.cats
        pred_annotations = []
        for idx in range(len(results)):
            result = results[idx]
            for segment_info in result['segments_info']:
                sem_label = segment_info['category_id']
                # convert sem_label to json label
                cat_id = label2cat[sem_label]
                segment_info['category_id'] = label2cat[sem_label]
                is_thing = categories[cat_id]['isthing']
                segment_info['isthing'] = is_thing
            pred_annotations.append(result)
        pan_json_results = dict(annotations=pred_annotations)
        json_filename = f'{outfile_prefix}.panoptic.json'
        dump(pan_json_results, json_filename)
        return json_filename, (
            self.seg_out_dir
            if self.tmp_dir is None else tempfile.gettempdir())

    def print_panoptic_table(self,
                             results: dict,
                             logger: Optional[Union['MMLogger', str]] = None) -> None:
        """Print the panoptic evaluation results table.

        Args:
            results(dict): The Panoptic results with the
                following keys:

                - pq_results(dict): The Panoptic Quality results.
                - classwise_results(dict, optional): The classwise Panoptic Quality.
                    results. The keys are class names and the values are metrics.
                    Defaults to None.
                - parse_results(dict): The parsed Panoptic Quality results.
            logger (:obj:`MMLogger` | str, optional): Logger used for printing
                related information during evaluation. Default: None.
        """
        pq_results = results['pq_results']
        classwise_results = results['classwise_results']
        headers = ['', 'PQ', 'SQ', 'RQ', 'categories']
        data = [headers]
        for name in ['All', 'Things', 'Stuff']:
            numbers = [
                f'{(pq_results[name][k] * 100):0.3f}' for k in ['pq', 'sq', 'rq']
            ]
            row = [name] + numbers + [pq_results[name]['n']]
            data.append(row)
        table = AsciiTable(data)
        print_log('Panoptic Evaluation Results:\n' +
                  table.table, logger=logger)

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

    def evaluate(self, *args, **kwargs) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        preds, gts = zip(*self._results)

        eval_results = OrderedDict()
        if self.format_only:
            _ = self.results2json(preds, self.outfile_prefix)
            logger.info('results are saved in '
                        f'{osp.dirname(self.outfile_prefix)}')
            return eval_results

        if self.ann_file is None:
            coco_json_path, gt_folder = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=self.outfile_prefix)

            self.ann_file = coco_json_path
        results = self.compute(*args, **kwargs)
        self.print_panoptic_table(results, logger=logger)
