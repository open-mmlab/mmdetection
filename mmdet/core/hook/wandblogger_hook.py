# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook

from mmdet.core import DistEvalHook, EvalHook


@HOOKS.register_module()
class MMDetWandbHook(WandbLoggerHook):
    """MMDetWandbHook logs metrics, saves model checkpoints as W&B Artifact,
    and logs model prediction as interactive W&B Tables.

    - Metrics: The MMDetWandbHook will automatically log training
        and validation metrics.

    - Checkpointing: If `log_checkpoint` is True, the checkpoint saved at
        every checkpoint interval will be saved as W&B Artifacts.
        This depends on the `CheckpointHook` whose priority is more
        than `MMDetWandbHook`. Please refer to
        https://docs.wandb.ai/guides/artifacts/model-versioning
        to learn more about model versioning with W&B Artifacts.

    - Checkpoint Metadata: If `log_checkpoint_metadata` is True, every
        checkpoint artifact will have a metadata associated with it.
        The metadata contains the evaluation metrics computed on validation
        data with that checkpoint along with the current epoch. If True, it
        also marks the checkpoint version with the best evaluation metric with
        a 'best' alias. You can choose the best checkpoint in the W&B Artifacts
        UI using this. It depends on `EvalHook` whose priority is more
        than MMDetWandbHook.

    - Evaluation: At every evaluation interval, the `MMDetWandbHook` logs the
        model prediction as interactive W&B Tables. The number of samples
        logged is given by `num_eval_images`. Currently, the `MMDetWandbHook`
        logs the predicted bounding boxes along with the ground truth at every
        evaluation interval. This depends on the `EvalHook` whose priority is
        more than `MMDetWandbHook`. Also note that the data is just logged once
        and subsequent evaluation tables uses reference to the logged data
        to save memory usage. Please refer to
        https://docs.wandb.ai/guides/data-vis to learn more about W&B Tables.

    For more details check out W&B's MMDetection docs:
    https://docs.wandb.ai/guides/integrations/mmdetection

    ```
    Example:
        log_config = dict(
            interval=10,
            hooks=[
                dict(type='MMDetWandbHook',
                     init_kwargs={
                         'entity': WANDB_ENTITY,
                         'project': WANDB_PROJECT_NAME
                     },
                     interval=10,
                     log_checkpoint=True,
                     log_checkpoint_metadata=True,
                     num_eval_images=100)
            ])
    ```

    Args:
        init_kwargs (dict): A dict passed to wandb.init to initialize
            a W&B run. Please refer to https://docs.wandb.ai/ref/python/init
            for possible key-value pairs.
        interval (int): Logging interval (every k iterations).
            Default 10.
        log_checkpoint (bool): Save the checkpoint at every checkpoint interval
            as W&B Artifacts. Use this for model versioning where each version
            is a checkpoint.
            Default: False
        log_checkpoint_metadata (bool): Log the evaluation metrics computed
            on the validation data with the checkpoint, along with current
            epoch as a metadata to that checkpoint.
            Default: True
        num_eval_images (int): Number of validation images to be logged.
            Default: 100
    """

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 log_checkpoint=False,
                 log_checkpoint_metadata=False,
                 num_eval_images=100,
                 **kwargs):
        super(MMDetWandbHook, self).__init__(init_kwargs, interval, **kwargs)

        self.log_checkpoint = log_checkpoint
        self.log_checkpoint_metadata = log_checkpoint_metadata
        self.num_eval_images = num_eval_images
        self.log_evaluation = True
        self.log_eval_metrics = True
        self.best_score = 0
        self.val_step = 0

    @master_only
    def before_run(self, runner):
        super(MMDetWandbHook, self).before_run(runner)
        self.cfg = self.wandb.config
        # Check if configuration is passed to wandb.
        if len(dict(self.cfg)) == 0:
            warnings.warn(
                'To log mmdetection Config, '
                'pass it to init_kwargs of MMDetWandbHook.', UserWarning)

        # Check if EvalHook and CheckpointHook are available.
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
            if isinstance(hook, (EvalHook, DistEvalHook)):
                self.eval_hook = hook

        # If CheckpointHook is not available turn off log_checkpoint.
        if getattr(self, 'ckpt_hook', None) is None:
            self.log_checkpoint = False
            warnings.warn('To use log_checkpoint turn use '
                          'CheckpointHook.', UserWarning)

        # If EvalHook/DistEvalHook is not present set
        # num_eval_images to zero.
        try:
            self.val_dataloader = self.eval_hook.dataloader
            self.val_dataset = self.val_dataloader.dataset
        except AttributeError:
            self.num_eval_images = 0
            self.log_checkpoint_metadata = False
            self.log_eval_metrics = False
            warnings.warn(
                'To log num_eval_images turn validate '
                'to True in train_detector.', UserWarning)

        # If num_eval_images is greater than zero, create
        # and log W&B table for validation data.
        if self.num_eval_images > 0:
            # Initialize data table
            self._init_data_table()
            # Add data to the table
            self._add_ground_truth()
            # Log ground truth data
            if self.log_evaluation:
                self._log_data_table()

        # Define a custom x-axes for validation metrics.
        if self.log_eval_metrics:
            self.wandb.define_metric('val/val_step')
            self.wandb.define_metric('val/*', step_metric='val/val_step')

    @master_only
    def after_train_epoch(self, runner):
        if self.log_eval_metrics:
            if self.eval_hook.by_epoch:
                if self.every_n_epochs(
                        runner,
                        self.eval_hook.interval) or self.is_last_epoch(runner):
                    results = self.eval_hook.results
                    eval_results = self.val_dataset.evaluate(
                        results, logger='silent')
                    for key, val in eval_results.items():
                        if isinstance(val, str):
                            continue
                        self.wandb.log({f'val/{key}': val}, commit=False)
                    self.wandb.log({'val/val_step': self.val_step})
                    self.val_step += 1

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

        if self.num_eval_images > 0 and self.log_evaluation:
            if self.eval_hook.by_epoch and self.eval_hook._should_evaluate(
                    runner):
                results = self.eval_hook.results
                # Initialize evaluation table
                self._init_pred_table()
                # Log predictions
                self._log_predictions(results, runner.epoch + 1)
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
        """Log model checkpoint as  W&B Artifact.

        Args:
            path_to_model (str): Path where model checkpoints are saved.
            epoch (int): The current epoch.
            aliases (list): List of the aliases associated with this artifact.
            metadata (dict, optional): Metadata associated with this artifact.
        """
        model_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_model', type='model', metadata=metadata)
        model_artifact.add_file(f'{path_to_model}/epoch_{epoch+1}.pth')
        self.wandb.log_artifact(model_artifact, aliases=aliases)

    def _get_ckpt_metadata(self, runner):
        """Get model checkpoint metadata."""
        if self.ckpt_hook.interval == self.eval_hook.interval:
            results = self.eval_hook.results
        else:
            runner.logger.info(
                f'Evaluating for model checkpoint at epoch '
                f'{runner.epoch+1} which will be saved as W&B Artifact.')
            if isinstance(self.eval_hook, EvalHook):
                from mmdet.apis import single_gpu_test
                results = single_gpu_test(
                    runner.model, self.val_dataloader, show=False)
            elif isinstance(self.eval_hook, DistEvalHook):
                from mmdet.apis import multi_gpu_test
                results = multi_gpu_test(
                    runner.model, self.val_dataloader, gpu_collect=True)

        eval_results = self.val_dataset.evaluate(results, logger='silent')
        metadata = dict(epoch=runner.epoch + 1, **eval_results)
        return metadata

    def _is_best_ckpt(self, metadata):
        """Check if the current checkpoint has the best metric score.

        Args:
            metadata (dict): Metadata associated with the checkpoint.

        Returns:
            bool: Returns True, if the checkpoint has the best metric score.
        """
        keys = list(metadata.keys())
        map_metrics = [key for key in keys if 'mAP' in key]
        ar_metrics = [key for key in keys if 'AR' in key]
        if len(map_metrics) > 0:
            map_score = metadata.get(map_metrics[0], None)
            return self._is_best_score(map_score)
        elif len(ar_metrics) > 0:
            ar_score = metadata.get(ar_metrics[0], None)
            return self._is_best_score(ar_score)
        else:
            return False

    def _is_best_score(self, score):
        if score is None:
            return

        if score > self.best_score:
            self.best_score = score
            return True
        else:
            return False

    def _init_data_table(self):
        """Initialize the W&B Tables for validation data."""
        columns = ['image_name', 'image']
        self.data_table = self.wandb.Table(columns=columns)

    def _init_pred_table(self):
        """Initialize the W&B Tables for model evaluation."""
        columns = ['epoch', 'image_name', 'ground_truth', 'prediction'] + list(
            self.class_id_to_label.values())
        self.eval_table = self.wandb.Table(columns=columns)

    def _add_ground_truth(self):
        # Get image loading pipeline
        from mmdet.datasets.pipelines import LoadImageFromFile
        transforms = self.val_dataset.pipeline.transforms
        for transform in transforms:
            if isinstance(transform, LoadImageFromFile):
                img_loader = transform
        if 'img_loader' not in locals():
            warnings.warn(
                'LoadImageFromFile is required to add images '
                'to W&B Tables.', UserWarning)
            self.log_evaluation = False

        # Determine the number of samples to be logged.
        num_total_images = len(self.val_dataset)
        if self.num_eval_images > num_total_images:
            warnings.warn(
                'The num_eval_images is greater than the total number '
                'of validation samples. The complete validation set '
                'will be logged.', UserWarning)
        self.num_eval_images = min(self.num_eval_images, num_total_images)

        classes = self.val_dataset.get_classes()
        self.class_id_to_label = {id: name for id, name in enumerate(classes)}
        self.class_set = self.wandb.Classes([{
            'id': id,
            'name': name
        } for id, name in self.class_id_to_label.items()])

        img_prefix = self.val_dataset.img_prefix

        for idx in range(self.num_eval_images):
            img_info = self.val_dataset.data_infos[idx]
            img_meta = img_loader(
                dict(img_info=img_info, img_prefix=img_prefix))

            # Get image and convert from BGR to RGB
            image = img_meta['img'][..., ::-1]
            image_name = img_info['filename']

            data_ann = self.val_dataset.get_ann_info(idx)
            bboxes = data_ann['bboxes']
            labels = data_ann['labels']

            assert len(bboxes) == len(labels)
            box_data = self._get_wandb_bboxes(bboxes, labels, mode='gt')

            boxes = {
                'ground_truth': {
                    'box_data': box_data,
                    'class_labels': self.class_id_to_label
                }
            }

            self.data_table.add_data(
                image_name,
                self.wandb.Image(image, boxes=boxes, classes=self.class_set))

    def _log_predictions(self, results, epoch):
        table_idxs = self.data_table_ref.get_index()
        assert len(table_idxs) == self.num_eval_images

        for ndx in table_idxs:
            result = results[ndx]
            assert len(result) == len(self.class_id_to_label)

            result_box_data = []
            class_scores = []
            for label_id, bboxes in enumerate(result):
                if len(bboxes) != 0:
                    labels = [label_id] * len(bboxes)
                    box_data, class_score, count = self._get_wandb_bboxes(
                        bboxes, labels, mode='pred')
                    result_box_data.extend(box_data)
                    class_scores.append(class_score / (count + 1e-6))
                else:
                    class_scores.append(0)

            boxes = {
                'predictions': {
                    'box_data': result_box_data,
                    'class_labels': self.class_id_to_label
                }
            }

            self.eval_table.add_data(
                epoch, self.data_table_ref.data[ndx][0],
                self.data_table_ref.data[ndx][1],
                self.wandb.Image(
                    self.data_table_ref.data[ndx][1],
                    boxes=boxes,
                    classes=self.class_set), *tuple(class_scores))

    def _get_wandb_bboxes(self, bboxes, labels, mode='gt'):  # pred
        """Get list of structured dict for logging bounding boxes to W&B.

        Args:
            bboxes (list): List of bounding box coordinates in
                        (minX, minY, maxX, maxY) format.
            labels (int): List of label ids.
            mode (str): Set it to 'gt' for ground truth and 'pred'
                for logging prediction results.

        Returns:
            List[dict]: List of structured dict required for
                logging that bounding boxes to W&B.
        """
        box_data = []

        if mode == 'pred':
            class_score = 0
            count = 0

        for bbox, label in zip(bboxes, labels):
            if len(bbox) == 5 and mode == 'pred':
                confidence = float(bbox[4])
                if confidence > 0.3:
                    class_score += confidence
                    count += 1
                    class_name = self.class_id_to_label[label]
                    box_caption = f'{class_name} {confidence:.2f}'
                else:
                    continue
            else:
                box_caption = str(self.class_id_to_label[label])

            position = dict(
                minX=int(bbox[0]),
                minY=int(bbox[1]),
                maxX=int(bbox[2]),
                maxY=int(bbox[3]))

            if not isinstance(label, int):
                label = int(label)

            box_data.append({
                'position': position,
                'class_id': label,
                'box_caption': box_caption,
                'domain': 'pixel'
            })

        if len(bboxes[0]) == 5 and mode == 'pred':
            return box_data, class_score, count

        return box_data

    def _log_data_table(self):
        """Log the W&B Tables for validation data as artifact and calls
        `use_artifact` on it so that the evaluation table can use the reference
        of already uploaded images.

        This allows the data to be uploaded just once.
        """
        data_artifact = self.wandb.Artifact('val', type='dataset')
        data_artifact.add(self.data_table, 'val_data')

        self.wandb.run.use_artifact(data_artifact)
        data_artifact.wait()

        self.data_table_ref = data_artifact.get('val_data')

    def _log_eval_table(self):
        """Log the W&B Tables for model evaluation.

        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.
        """
        pred_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_pred', type='evaluation')
        pred_artifact.add(self.eval_table, 'eval_data')
        self.wandb.run.log_artifact(pred_artifact)
