# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
from mmengine.model import is_model_wrapper
from mmengine.runner import TestLoop, ValLoop
from mmengine.runner.amp import autocast

from mmdet.registry import LOOPS


@LOOPS.register_module()
class TeacherStudentValLoop(ValLoop):
    """Loop for validation of model teacher and student."""

    def run(self):
        """Launch validation for model teacher and student."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        model = self.runner.model
        if is_model_wrapper(model):
            model = model.module
        assert hasattr(model, 'teacher')
        assert hasattr(model, 'student')

        predict_on = model.semi_test_cfg.get('predict_on', None)
        multi_metrics = dict()
        for _predict_on in ['teacher', 'student']:
            model.semi_test_cfg['predict_on'] = _predict_on
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
            # compute metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            multi_metrics.update(
                {'/'.join((_predict_on, k)): v
                 for k, v in metrics.items()})
        model.semi_test_cfg['predict_on'] = predict_on

        self.runner.call_hook('after_val_epoch', metrics=multi_metrics)
        self.runner.call_hook('after_val')


@LOOPS.register_module()
class VideoValLoop(ValLoop):
    """Loop for validation of video model.

    The difference between this loop and ``ValLoop`` is that this loop does not
    pass the predictions into the evaluator util the whole video is processed.
    """

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)

        assert len(outputs) == 1, 'During inference, only support 1 batch '
        'size per gpu for now in video tasks.'

        track_data_sample = outputs[0]
        ori_video_len = track_data_sample.ori_video_length
        if len(track_data_sample) == ori_video_len:
            self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        else:
            frame_id = track_data_sample[0].frame_id
            if frame_id == 0:
                self.video_outputs = track_data_sample.clone()
            else:
                self.video_outputs.video_data_samples.append(
                    track_data_sample[0].clone())
                self.video_outputs.set_metainfo(
                    dict(video_length=self.video_outputs.video_length + 1))
                if frame_id == ori_video_len - 1:
                    self.evaluator.process(
                        data_samples=[self.video_outputs],
                        data_batch=data_batch)

        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class VideoTestLoop(TestLoop):
    """Loop for test of video model.

    The difference between this loop and ``ValLoop`` is that this loop does not
    pass the predictions into the evaluator util the whole video is processed.
    """

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)

        assert len(outputs) == 1, 'During inference, only support 1 batch '
        'size per gpu for now in video tasks.'

        track_data_sample = outputs[0]
        ori_video_len = track_data_sample.ori_video_length
        if len(track_data_sample) == ori_video_len:
            self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        else:
            frame_id = track_data_sample[0].frame_id
            if frame_id == 0:
                self.video_outputs = track_data_sample.clone()
            else:
                self.video_outputs.video_data_samples.append(
                    track_data_sample[0].clone())
                self.video_outputs.set_metainfo(
                    dict(video_length=self.video_outputs.video_length + 1))
                if frame_id == ori_video_len - 1:
                    self.evaluator.process(
                        data_samples=[self.video_outputs],
                        data_batch=data_batch)

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
