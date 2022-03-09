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
                 log_checkpoint_metadata=False,
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
        self.log_checkpoint_metadata = log_checkpoint_metadata
        self.log_evaluation = log_evaluation

        self.best_map_score = 0

    @master_only
    def before_run(self, runner):
        super(WandbLogger, self).before_run(runner)
        self.cfg = self.wandb.config
        if len(dict(self.cfg)) == 0:
            warnings.warn(
                'To log mmdetection Config,'
                'pass it to init_kwargs of WandbLogger.', UserWarning)

        for hook in runner.hooks:
            if isinstance(hook, EvalHook):
                eval_hook = hook
            if isinstance(hook, CheckpointHook):
                ckpt_hook = hook

        if 'ckpt_hook' in locals():
            self.ckpt_hook = ckpt_hook
        else:
            self.log_checkpoint = False

        if 'eval_hook' in locals():
            self.eval_hook = eval_hook
            self.val_dataloader = eval_hook.dataloader
            self.val_dataset = self.val_dataloader.dataset
        else:
            self.log_evaluation = False
            warnings.warn(
                'To use log_evaluation turn validate '
                'to True in train_detector.', UserWarning)

        if self.log_checkpoint:
            # Remove model checkpoints from previous run.
            files = glob.glob(f'{runner.work_dir}/*.pth')
            for f in files:
                os.remove(f)

        if self.log_evaluation:
            # Initialize data table
            self._init_data_table()
            # Add data to the table
            self._add_ground_truth()

    @master_only
    def after_train_epoch(self, runner):
        if self.log_checkpoint:
            if self.ckpt_hook.by_epoch:
                if self.every_n_epochs(runner, self.ckpt_hook.interval) or (
                        self.ckpt_hook.save_last
                        and self.is_last_epoch(runner)):
                    if self.log_checkpoint_metadata and self.eval_hook:
                        metadata = self._get_ckpt_metadata(runner)
                        aliases = [f'epoch_{runner.epoch+1}', 'latest']
                        if self._is_best_ckpt(metadata):
                            aliases.append('best')
                        self._log_ckpt_as_artifact(self.ckpt_hook.out_dir,
                                                   runner.epoch, aliases,
                                                   metadata)
                    else:
                        aliases = [f'epoch_{runner.epoch+1}', 'latest']
                        self._log_ckpt_as_artifact(self.ckpt_hook.out_dir,
                                                   runner.epoch, aliases)

        if self.log_evaluation:
            if self.eval_hook.by_epoch:
                if self.every_n_epochs(
                        runner,
                        self.eval_hook.interval) or self.is_last_epoch(runner):
                    runner.logger.info(
                        f'Running inference at epoch {runner.epoch+1} for W&B '
                        f'evaluation table which will be saved as W&B Tables.')
                    from mmdet.apis import single_gpu_test
                    self.results = single_gpu_test(
                        runner.model,
                        self.val_dataloader,
                        show=False,
                        rescale=False)
                    # Initialize evaluation table
                    self._init_pred_table()
                    # Log predictions
                    self._log_predictions(runner.epoch + 1)
                    # Log the table
                    self._log_eval_table()

    @master_only
    def after_run(self, runner):
        self.wandb.finish()

    def _log_ckpt_as_artifact(self,
                              path_to_model,
                              epoch,
                              aliases,
                              metadata=None):
        model_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_model', type='model', metadata=metadata)
        model_artifact.add_file(f'{path_to_model}/epoch_{epoch+1}.pth')
        self.wandb.log_artifact(model_artifact, aliases=aliases)

    def _get_ckpt_metadata(self, runner):
        runner.logger.info(
            f'Evaluating for model checkpoint at epoch '
            f'{runner.epoch+1} which will be saved as W&B Artifact.')
        from mmdet.apis import single_gpu_test
        results = single_gpu_test(
            runner.model, self.val_dataloader, show=False, rescale=True)
        eval_results = self.val_dataset.evaluate(results)
        map_score = eval_results.get('mAP', None)
        if map_score:
            metadata = dict(epoch=runner.epoch + 1, map_score=map_score)
            return metadata
        else:
            warnings.warn(
                'Set eval metric to mAP to get metadata for checkpoint.',
                UserWarning)
            return {}

    def _is_best_ckpt(self, metadata):
        map_score = metadata.get('map_score', None)
        if map_score:
            if map_score > self.best_map_score:
                self.best_map_score = map_score
                return True
            else:
                return False

    def _init_data_table(self):
        columns = ['image_name', 'image']
        self.data_table = self.wandb.Table(columns=columns)

    def _init_pred_table(self):
        columns = ['epoch', 'image_name', 'ground_truth', 'prediction'] + list(
            self.class_id_to_label.values())
        self.eval_table = self.wandb.Table(columns=columns)

    def _add_ground_truth(self):
        num_images = len(self.val_dataset)
        assert num_images == len(self.val_dataset.data_infos)

        classes = self.val_dataset.get_classes()
        self.class_id_to_label = {id: name for id, name in enumerate(classes)}
        self.class_set = self.wandb.Classes([{
            'id': id,
            'name': name
        } for id, name in self.class_id_to_label.items()])

        for idx in range(num_images):
            data_sample = self.val_dataset.prepare_test_img(idx)
            data_ann = self.val_dataset.get_ann_info(idx)

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

    def _log_predictions(self, epoch):
        table_idxs = self.data_table.get_index()
        assert len(self.results) == len(self.val_dataset) == len(table_idxs)

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
                epoch, self.data_table.data[ndx][0],
                self.data_table.data[ndx][1],
                self.wandb.Image(
                    self.data_table.data[ndx][1],
                    boxes=boxes,
                    classes=self.class_set), *tuple(class_scores))

    def _log_data_table(self):
        data_artifact = self.wandb.Artifact('val', type='dataset')
        data_artifact.add(self.data_table, 'val_data')
        self.wandb.run.log_artifact(self.data_artifact)

    def _log_eval_table(self):
        pred_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_pred', type='evaluation')
        pred_artifact.add(self.eval_table, 'eval_data')
        self.wandb.run.log_artifact(pred_artifact)
