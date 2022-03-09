# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import warnings

from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook

from mmdet.core import EvalHook


@HOOKS.register_module()
class WandbLogger(WandbLoggerHook):

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 commit=True,
                 by_epoch=True,
                 with_step=True,
                 log_checkpoint=False,
                 log_evaluation=False):
        super(WandbLogger, self).__init__(
            init_kwargs,
            interval,
            ignore_last,
            reset_flag,
            commit,
            by_epoch,
            with_step,
        )

        self.log_checkpoint = log_checkpoint
        self.log_evaluation = log_evaluation

    @master_only
    def before_run(self, runner):
        super(WandbLogger, self).before_run(runner)
        self.cfg = self.wandb.config
        if len(dict(self.cfg)) == 0:
            warnings.warn(
                'To log mmdetection Config,\
                pass it to init_kwargs of WandbLogger.', UserWarning)

        for hook in runner.hooks:
            if isinstance(hook, EvalHook):
                eval_hook = hook
            if isinstance(hook, CheckpointHook):
                ckpt_hook = hook

        if 'ckpt_hook' in locals():
            pass
        else:
            self.log_checkpoint = False

        if 'eval_hook' in locals():
            self.dataloader = eval_hook.dataloader
            self.dataset = self.dataloader.dataset
        else:
            self.log_evaluation = False
            warnings.warn(
                'To use log_evaluation turn validate\
                to True in train_detector.', UserWarning)

        if self.log_checkpoint:
            # Remove model checkpoint from previous run.
            files = glob.glob(f'{runner.work_dir}/*.pth')
            for f in files:
                os.remove(f)

        if self.log_evaluation:
            # Initialize data table
            self._init_data_table()
            # Add data to the table
            self._add_data_table()

    @master_only
    def after_train_epoch(self, runner):
        # print(runner.hooks)
        pass

    @master_only
    def after_run(self, runner):
        from mmdet.apis import single_gpu_test
        self.results = single_gpu_test(
            runner.model, self.dataloader, show=False, rescale=False)

        # Model evaluation
        if self.log_evaluation:
            # Initialize evaluation table
            self._init_pred_table(f'run_{self.wandb.run.id}_pred')
            # Log predictions
            self._log_predictions()
            # Log the table
            self._log_table()

        self.wandb.finish()

    def _init_data_table(self, name='val'):
        self.data_artifact = self.wandb.Artifact(name, type='dataset')
        columns = ['image_name', 'image']
        self.data_table = self.wandb.Table(columns=columns)

    def _init_pred_table(self, name):
        self.pred_artifact = self.wandb.Artifact(name, type='evaluation')
        columns = ['image_name', 'ground_truth', 'prediction'] + list(
            self.class_id_to_label.values())
        self.eval_table = self.wandb.Table(columns=columns)

    def _add_data_table(self):
        num_images = len(self.dataset)
        assert num_images == len(self.dataset.data_infos)

        classes = self.dataset.get_classes()
        self.class_id_to_label = {id: name for id, name in enumerate(classes)}
        self.class_set = self.wandb.Classes([{
            'id': id,
            'name': name
        } for id, name in self.class_id_to_label.items()])

        for idx in range(num_images):
            data_sample = self.dataset.prepare_test_img(idx)
            data_ann = self.dataset.get_ann_info(idx)

            img_meta = data_sample['img_metas'][0].data
            image = data_sample['img'][0].data
            bboxes = data_ann['bboxes']
            labels = data_ann['labels']
            scale_factor = img_meta['scale_factor']

            image_name = img_meta['ori_filename']
            # permute to get channel last configuration, followed by bgr -> rgb
            image = image.permute(1, 2, 0).numpy()[..., ::-1]

            box_data = []
            assert len(bboxes) == len(labels)
            for bbox, label in zip(bboxes, labels):
                position = dict(
                    minX=int(bbox[0] * scale_factor[0]),
                    minY=int(bbox[1] * scale_factor[1]),
                    maxX=int(bbox[2] * scale_factor[2]),
                    maxY=int(bbox[3] * scale_factor[3]))
                box_data.append({
                    'position': position,
                    'class_id': int(label),
                    'box_caption': classes[label],
                    'domain': 'pixel'
                })

            boxes = {
                'ground_truth': {
                    'box_data': box_data,
                    'class_labels': self.class_id_to_label
                }
            }
            self.data_table.add_data(
                image_name,
                self.wandb.Image(image, boxes=boxes, classes=self.class_set))

    def _log_predictions(self):
        table_idxs = self.data_table.get_index()
        assert len(self.results) == len(self.dataset) == len(table_idxs)

        for ndx in table_idxs:
            result = self.results[ndx]
            assert len(result) == len(self.class_id_to_label)

            box_data = []
            class_scores = []
            for label_id, bbox_scores in enumerate(result):
                if len(bbox_scores) != 0:
                    class_score = 0
                    count = 0
                    for bbox_score in bbox_scores:
                        confidence = float(bbox_score[4])
                        if confidence > 0.3:
                            class_score += confidence
                            count += 1
                            class_name = self.class_id_to_label[label_id]

                            position = dict(
                                minX=int(bbox_score[0]),
                                minY=int(bbox_score[1]),
                                maxX=int(bbox_score[2]),
                                maxY=int(bbox_score[3]))

                            box_data.append({
                                'position': position,
                                'class_id': label_id,
                                'box_caption':
                                f'{class_name}@{confidence:.2f}',
                                'domain': 'pixel'
                            })

                    class_scores.append(class_score / (count + 1e-6))
                else:
                    class_scores.append(0)

            boxes = {
                'predictions': {
                    'box_data': box_data,
                    'class_labels': self.class_id_to_label
                }
            }
            self.eval_table.add_data(
                self.data_table.data[ndx][0], self.data_table.data[ndx][1],
                self.wandb.Image(
                    self.data_table.data[ndx][1],
                    boxes=boxes,
                    classes=self.class_set), *tuple(class_scores))

    def _log_table(self):
        self.data_artifact.add(self.data_table, 'val_data')
        self.pred_artifact.add(self.eval_table, 'eval_data')

        self.wandb.run.log_artifact(self.data_artifact)
        self.wandb.run.log_artifact(self.pred_artifact)
