# Modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/cityscapes.py # noqa
# and https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa

import glob
import multiprocessing as mp
import os
import os.path as osp
import tempfile

import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as CSEval
import cityscapesscripts.helpers.labels as CSLabels
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmcv.runner import get_dist_info

from mmdet.utils import print_log
from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class CityscapesDataset(CustomDataset):
    CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    def load_annotations(self, gt_dir):
        self.cat_ids = [
            label.id for label in CSLabels.labels
            if label.hasInstances and not label.ignoreInEval
        ]
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        suffix = 'leftImg8bit.png'
        files = []
        for img_file in glob.glob(osp.join(self.img_prefix, '**/*.png')):
            assert img_file.endswith(suffix), img_file
            inst_file = gt_dir + img_file[
                len(self.img_prefix):-len(suffix)] + 'gtFine_instanceIds.png'
            # Note that labelIds are not converted to trainId for seg map
            segm_file = gt_dir + img_file[
                len(self.img_prefix):-len(suffix)] + 'gtFine_labelIds.png'
            files.append((img_file, inst_file, segm_file))
        assert len(files), 'No images found in {}'.format(self.img_prefix)
        # img_infos = [self._load_img_info(_) for _ in files]
        print_log('Loading annotation images')
        with mmcv.Timer(print_tmpl='It took {}s to load annotation.'):
            pool = mp.Pool(
                processes=max(mp.cpu_count() // get_dist_info()[1] // 2, 4))
            img_infos = pool.map(self._load_img_info, files)
        print_log('Loaded {} images from {}'.format(
            len(img_infos), self.img_prefix))
        return img_infos

    def _load_img_info(self, files):
        img_file, inst_file, segm_file = files
        inst_img = mmcv.imread(inst_file, 'unchanged')
        # ids < 24 are stuff labels (filtering them first is about 5% faster)
        unique_inst_ids = np.unique(inst_img[inst_img >= 24])
        anno_info = []
        for inst_id in unique_inst_ids:
            # For non-crowd annotations, inst_id // 1000 is the label_id
            # Crowd annotations have <1000 instance ids
            label_id = inst_id // 1000 if inst_id >= 1000 else inst_id
            label = CSLabels.id2label[label_id]
            if not label.hasInstances or label.ignoreInEval:
                continue

            iscrowd = inst_id < 1000
            category_id = label.id
            mask = np.asarray(inst_img == inst_id, dtype=np.uint8, order='F')
            # encode mask to RLE to save memory
            mask_rle = maskUtils.encode(mask[:, :, None])[0]
            inds = np.nonzero(mask)
            ymin, ymax = inds[0].min(), inds[0].max()
            xmin, xmax = inds[1].min(), inds[1].max()
            bbox = (xmin, ymin, xmax, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue

            anno = dict(
                iscrowd=iscrowd,
                category_id=category_id,
                bbox=bbox,
                segmentation=mask_rle)
            anno_info.append(anno)
        img_info = dict(
            # remove img_prefix for filename
            filename=img_file.replace(self.img_prefix, ''),
            image_id=osp.basename(img_file),
            height=inst_img.shape[0],
            width=inst_img.shape[1],
            anno_info=anno_info,
            segm_file=segm_file)
        return img_info

    def get_ann_info(self, idx):
        return self._parse_ann_info(self.img_infos[idx])

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt:
                if len(img_info['anno_info']) == 0 or \
                        all([_['iscrowd'] for _ in img_info['anno_info']]):
                    continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info):
        """Parse bbox and mask annotation.

        Args:
            img_info (dict): Image info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map.
                "masks" are already decoded into binary masks.
        """
        ann_info = img_info['anno_info']
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

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
            masks=gt_masks_ann,
            seg_map=img_info['segm_file'])

        return ann

    def results2txt(self, results, outfile_prefix):
        """Dump the detection results to a txt file.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files.
                If the prefix is "somepath/xxx",
                the txt files will be named "somepath/xxx.txt".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and
                values are corresponding filenames.
        """
        os.makedirs(outfile_prefix, exist_ok=True)
        for idx, result in enumerate(results):
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]
            pred_txt = osp.join(outfile_prefix, basename + '_pred.txt')

            bbox_result, segm_result = result
            bboxes = np.vstack(bbox_result)
            segms = mmcv.concat_list(segm_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            assert len(bboxes) == len(segms) == len(labels)
            num_instances = len(bboxes)
            with open(pred_txt, 'w') as fout:
                for i in range(num_instances):
                    pred_class = labels[i]
                    classes = self.CLASSES[pred_class]
                    class_id = CSLabels.name2label[classes].id
                    score = bboxes[i, -1]
                    mask = maskUtils.decode(segms[i]).astype(np.uint8)
                    png_filename = osp.join(
                        outfile_prefix,
                        basename + '_{}_{}.png'.format(i, classes))
                    mmcv.imwrite(mask, png_filename)
                    fout.write('{} {} {}\n'.format(
                        osp.basename(png_filename), class_id, score))

    def evaluate(self,
                 results,
                 metric='segm',
                 logger=None,
                 txtfile_prefix=None):
        """Evaluation in Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            txtfile_prefix (str | None):

        Returns:
            dict[str: float]
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if txtfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            txtfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        print_log('Temp: {}'.format(txtfile_prefix), logger=logger)
        self.results2txt(results, txtfile_prefix)

        eval_results = {}
        print_log(
            'Evaluating results under {} ...'.format(txtfile_prefix),
            logger=logger)

        # set global states in cityscapes evaluation API
        CSEval.args.cityscapesPath = os.path.join(self.ann_file, '../..')
        CSEval.args.predictionPath = os.path.abspath(txtfile_prefix)
        CSEval.args.predictionWalk = None
        CSEval.args.JSONOutput = False
        CSEval.args.colorized = False
        CSEval.args.gtInstancesFile = os.path.join(txtfile_prefix,
                                                   'gtInstances.json')
        CSEval.args.groundTruthSearch = os.path.join(
            self.ann_file, '*/*_gtFine_instanceIds.png')

        groundTruthImgList = glob.glob(CSEval.args.groundTruthSearch)
        assert len(groundTruthImgList), \
            'Cannot find ground truth images in {}.'.format(
                CSEval.args.groundTruthSearch)
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(CSEval.getPrediction(gt, CSEval.args))
        CSEval_results = CSEval.evaluateImgLists(predictionImgList,
                                                 groundTruthImgList,
                                                 CSEval.args)['averages']

        eval_results['mAP'] = CSEval_results['allAp'] * 100
        eval_results['AP@50'] = CSEval_results['allAp50%'] * 100
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
