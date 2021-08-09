import os
from collections import defaultdict

import mmcv
import numpy as np
from mmcv.utils import print_log

from .api_wrappers import COCO
from .builder import DATASETS
from .coco import CocoDataset

try:
    import panopticapi
    from panopticapi.evaluation import pq_compute_multi_core, VOID
    from panopticapi.utils import id2rgb
except ImportError:
    panopticapi = None
    pq_compute_multi_core = None
    id2rgb = None
    VOID = None

__all__ = ['CocoPanopticDataset']

# A custom value to distinguish instance ID and category ID; need to
# be greater than the number of categories.
# For a pixel in the panoptic result map:
#   pan_id = ins_id * INSTANCE_OFFSET + cat_id
INSTANCE_OFFSET = 1000


class COCOPanoptic(COCO):
    """This wrapper is for loading the panoptic style annotation file.

    The format is shown in the CocoPanopticDataset class.

    Args:
        annotation_file (str): Path of annotation file.
    """

    def __init__(self, annotation_file=None):
        if panopticapi is None:
            raise RuntimeError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')

        super(COCOPanoptic, self).__init__(annotation_file)

    def createIndex(self):
        # create index
        print('creating index...')
        # anns stores 'segment_id -> annotation'
        anns, cats, imgs = {}, {}, {}
        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann, img_info in zip(self.dataset['annotations'],
                                     self.dataset['images']):
                img_info['segm_file'] = ann['file_name']
                for seg_ann in ann['segments_info']:
                    # to match with instance.json
                    seg_ann['image_id'] = ann['image_id']
                    seg_ann['height'] = img_info['height']
                    seg_ann['width'] = img_info['width']
                    img_to_anns[ann['image_id']].append(seg_ann)
                    # segment_id is not unique in coco dataset orz...
                    if seg_ann['id'] in anns.keys():
                        anns[seg_ann['id']].append(seg_ann)
                    else:
                        anns[seg_ann['id']] = [seg_ann]

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    cat_to_imgs[seg_ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats

    def load_anns(self, ids=[]):
        """Load anns with the specified ids.

        self.anns is a list of annotation lists instead of a
        list of annotations.

        Args:
            ids (int array): integer ids specifying anns

        Returns:
            anns (object array): loaded ann objects
        """
        anns = []

        if hasattr(ids, '__iter__') and hasattr(ids, '__len__'):
            # self.anns is a list of annotation lists instead of
            # a list of annotations
            for id in ids:
                anns += self.anns[id]
            return anns
        elif type(ids) == int:
            return self.anns[ids]


@DATASETS.register_module()
class CocoPanopticDataset(CocoDataset):
    """Coco dataset for Panoptic segmentation.

    The annotation format is shown as follows. The `ann` field is optional
    for testing.

    .. code-block:: none

        [
            {
                'filename': f'{image_id:012}.png',
                'image_id':9
                'segments_info': {
                    [
                        {
                            'id': 8345037, (segment_id in panoptic png,
                                            convert from rgb)
                            'category_id': 51,
                            'iscrowd': 0,
                            'bbox': (x1, y1, w, h),
                            'area': 24315,
                            'segmentation': list,(coded mask)
                        },
                        ...
                    }
                }
            },
            ...
        ]
    """
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        ' truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
        'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
        'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
        'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
        'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
        'wall-wood', 'water-other', 'window-blind', 'window-other',
        'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
        'cabinet-merged', 'table-merged', 'floor-other-merged',
        'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
        'paper-merged', 'food-other-merged', 'building-other-merged',
        'rock-merged', 'wall-other-merged', 'rug-merged'
    ]
    THING_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    STUFF_CLASSES = [
        'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain',
        'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house',
        'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
        'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
        'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
        'wall-wood', 'water-other', 'window-blind', 'window-other',
        'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
        'cabinet-merged', 'table-merged', 'floor-other-merged',
        'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
        'paper-merged', 'food-other-merged', 'building-other-merged',
        'rock-merged', 'wall-other-merged', 'rug-merged'
    ]

    def load_annotations(self, ann_file):
        """Load annotation from COCO Panoptic style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCOPanoptic(ann_file)
        self.cat_ids = self.coco.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.categories = self.coco.cats
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            info['segm_file'] = info['filename'].replace('jpg', 'png')
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        # filter out unmatched images
        ann_info = [i for i in ann_info if i['image_id'] == img_id]
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse annotations and load panoptic ground truths.

        Args:
            img_info (int): Image info of an image.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_mask_infos = []

        for i, ann in enumerate(ann_info):
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            category_id = ann['category_id']
            contiguous_cat_id = self.cat2label[category_id]

            is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
            if is_thing:
                is_crowd = ann.get('iscrowd', False)
                if not is_crowd:
                    gt_bboxes.append(bbox)
                    gt_labels.append(contiguous_cat_id)
                else:
                    gt_bboxes_ignore.append(bbox)
                    is_thing = False

            mask_info = {
                'id': ann['id'],
                'category': contiguous_cat_id,
                'is_thing': is_thing
            }
            gt_mask_infos.append(mask_info)

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

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_mask_infos,
            seg_map=img_info['segm_file'])

        return ann

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        ids_with_ann = []
        # check whether images have legal thing annotations.
        for lists in self.coco.anns.values():
            for item in lists:
                category_id = item['category_id']
                is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
                if not is_thing:
                    continue
                ids_with_ann.append(item['image_id'])
        ids_with_ann = set(ids_with_ann)

        valid_inds = []
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _pan2json(self, results, outfile_prefix):
        """Convert panoptic results to COCO panoptic json style."""
        label2cat = dict((v, k) for (k, v) in self.cat2label.items())
        pred_annotations = []
        outdir = os.path.join(os.path.dirname(outfile_prefix), 'panoptic')

        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            segm_file = self.data_infos[idx]['segm_file']
            pan = results[idx]

            pan_labels = np.unique(pan)
            segm_info = []
            for pan_label in pan_labels:
                sem_label = pan_label % INSTANCE_OFFSET
                # We reserve the length of self.CLASSES for VOID label
                if sem_label == len(self.CLASSES):
                    continue
                # convert sem_label to json label
                cat_id = label2cat[sem_label]
                is_thing = self.categories[cat_id]['isthing']
                mask = pan == pan_label
                area = mask.sum()
                segm_info.append({
                    'id': int(pan_label),
                    'category_id': cat_id,
                    'isthing': is_thing,
                    'area': int(area)
                })
            # evaluation script uses 0 for VOID label.
            pan[pan % INSTANCE_OFFSET == len(self.CLASSES)] = VOID
            pan = id2rgb(pan).astype(np.uint8)
            mmcv.imwrite(pan[:, :, ::-1], os.path.join(outdir, segm_file))
            record = {
                'image_id': img_id,
                'segments_info': segm_info,
                'file_name': segm_file
            }
            pred_annotations.append(record)
        pan_json_results = dict(annotations=pred_annotations)
        return pan_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the panoptic results to a COCO panoptic style json file.

        Args:
            results (dict): Testing results of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.panoptic.json"

        Returns:
            dict[str: str]: The key is 'panoptic' and the value is
                corresponding filename.
        """
        result_files = dict()
        pan_results = [result['pan_results'] for result in results]
        pan_json_results = self._pan2json(pan_results, outfile_prefix)
        result_files['panoptic'] = f'{outfile_prefix}.panoptic.json'
        mmcv.dump(pan_json_results, result_files['panoptic'])

        return result_files

    def evaluate_pan_json(self, result_files, outfile_prefix, logger=None):
        """Evaluate PQ according to the panoptic results json file."""
        imgs = self.coco.imgs
        gt_json = self.coco.img_ann_map  # image to annotations
        gt_json = [{
            'image_id': k,
            'segments_info': v,
            'file_name': imgs[k]['segm_file']
        } for k, v in gt_json.items()]
        pred_json = mmcv.load(result_files['panoptic'])
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

        gt_folder = self.seg_prefix
        pred_folder = os.path.join(os.path.dirname(outfile_prefix), 'panoptic')

        pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder,
                                        pred_folder, self.categories)

        eval_results = {}

        metrics = [('All', None), ('Things', True), ('Stuff', False)]
        pq_results = {}

        output = '\n'
        for name, isthing in metrics:
            pq_results[name], per_class_pq_results = pq_stat.pq_average(
                self.categories, isthing=isthing)
            if name == 'All':
                pq_results['per_class'] = per_class_pq_results
        output += ('{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}\n'.format(
            '', 'PQ', 'SQ', 'RQ', 'N'))
        output += ('-' * (10 + 7 * 4) + '\n')

        for name, _isthing in metrics:
            output += '{:10s}| {:5.2f}  {:5.2f}  {:5.2f} {:5d}\n'.format(
                name, 100 * pq_results[name]['pq'],
                100 * pq_results[name]['sq'], 100 * pq_results[name]['rq'],
                pq_results[name]['n'])
            eval_results[f'{name}_pq'] = pq_results[name]['pq'] * 100.0
            eval_results[f'{name}_sq'] = pq_results[name]['sq'] * 100.0
            eval_results[f'{name}_rq'] = pq_results[name]['rq'] * 100.0
        print_log(output, logger=logger)

        return eval_results

    def evaluate(self,
                 results,
                 metric='pq',
                 logger=None,
                 jsonfile_prefix=None,
                 **kwargs):
        """Evaluation in COCO Panoptic protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Only
                support 'pq' at present.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            dict[str, float]: COCO Panoptic style evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['pq']  # todo: support other metrics like 'bbox'
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = {}

        outfile_prefix = os.path.join(tmp_dir.name, 'results') \
            if tmp_dir is not None else jsonfile_prefix
        if 'pq' in metrics:
            eval_pan_results = self.evaluate_pan_json(result_files,
                                                      outfile_prefix, logger)
            eval_results.update(eval_pan_results)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
