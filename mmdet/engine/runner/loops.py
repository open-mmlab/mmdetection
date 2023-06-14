# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import is_model_wrapper
from mmengine.runner import ValLoop

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
