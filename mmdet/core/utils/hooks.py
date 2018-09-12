import os
import os.path as osp
import shutil
import time

import mmcv
import numpy as np
import torch
from mmcv.torchpack import Hook
from mmdet.datasets import collate
from mmdet.nn.parallel import scatter
from pycocotools.cocoeval import COCOeval

from ..eval import eval_recalls

__all__ = [
    'EmptyCacheHook', 'DistEvalHook', 'DistEvalRecallHook',
    'CocoDistEvalmAPHook'
]


class EmptyCacheHook(Hook):

    def before_epoch(self, runner):
        torch.cuda.empty_cache()

    def after_epoch(self, runner):
        torch.cuda.empty_cache()


class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        self.dataset = dataset
        self.interval = interval
        self.lock_dir = None

    def _barrier(self, rank, world_size):
        """Due to some issues with `torch.distributed.barrier()`, we have to
        implement this ugly barrier function.
        """
        if rank == 0:
            for i in range(1, world_size):
                tmp = osp.join(self.lock_dir, '{}.pkl'.format(i))
                while not (osp.exists(tmp)):
                    time.sleep(1)
            for i in range(1, world_size):
                tmp = osp.join(self.lock_dir, '{}.pkl'.format(i))
                os.remove(tmp)
        else:
            tmp = osp.join(self.lock_dir, '{}.pkl'.format(rank))
            mmcv.dump([], tmp)
            while osp.exists(tmp):
                time.sleep(1)

    def before_run(self, runner):
        self.lock_dir = osp.join(runner.work_dir, '.lock_map_hook')
        if runner.rank == 0:
            if osp.exists(self.lock_dir):
                shutil.rmtree(self.lock_dir)
            mmcv.mkdir_or_exist(self.lock_dir)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            device_id = torch.cuda.current_device()
            imgs_data = tuple(
                scatter(collate([data], samples_per_gpu=1), [device_id])[0])

            # compute output
            with torch.no_grad():
                result = runner.model(
                    *imgs_data,
                    return_loss=False,
                    return_bboxes=True,
                    rescale=True)
            results[idx] = result

            batch_size = runner.world_size
            for _ in range(batch_size):
                prog_bar.update()

        if runner.rank == 0:
            print('\n')
            self._barrier(runner.rank, runner.world_size)
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            self._barrier(runner.rank, runner.world_size)
        self._barrier(runner.rank, runner.world_size)

    def evaluate(self):
        raise NotImplementedError


class CocoEvalMixin(object):

    def _xyxy2xywh(self, bbox):
        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0] + 1,
            _bbox[3] - _bbox[1] + 1,
        ]

    def det2json(self, dataset, results):
        json_results = []
        for idx in range(len(dataset)):
            img_id = dataset.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self._xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = dataset.cat_ids[label]
                    json_results.append(data)
        return json_results

    def segm2json(self, dataset, results):
        json_results = []
        for idx in range(len(dataset)):
            img_id = dataset.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                bboxes = det[label]
                segms = seg[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self._xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = dataset.cat_ids[label]
                    segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    json_results.append(data)
        return json_results

    def proposal2json(self, dataset, results):
        json_results = []
        for idx in range(len(dataset)):
            img_id = dataset.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self._xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def results2json(self, dataset, results, out_file):
        if isinstance(results[0], list):
            json_results = self.det2json(dataset, results)
        elif isinstance(results[0], tuple):
            json_results = self.segm2json(dataset, results)
        elif isinstance(results[0], np.ndarray):
            json_results = self.proposal2json(dataset, results)
        else:
            raise TypeError('invalid type of results')
        mmcv.dump(json_results, out_file, file_format='json')


class DistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(DistEvalRecallHook, self).__init__(dataset)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # official coco evaluation is too slow, here we use our own
        # implementation, which may get slightly different results
        gt_bboxes = []
        for i in range(len(self.dataset)):
            img_id = self.dataset.img_ids[i]
            ann_ids = self.dataset.coco.getAnnIds(imgIds=img_id)
            ann_info = self.dataset.coco.loadAnns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes,
            results,
            self.proposal_nums,
            self.iou_thrs,
            print_summary=False)
        ar = recalls.mean(axis=1)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook, CocoEvalMixin):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0.json')
        self.results2json(self.dataset, results, tmp_file)

        res_types = ['bbox', 'segm'] if runner.model.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        cocoDt = cocoGt.loadRes(tmp_file)
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            field = '{}_mAP'.format(res_type)
            runner.log_buffer.output[field] = cocoEval.stats[0]
        runner.log_buffer.ready = True
        os.remove(tmp_file)
