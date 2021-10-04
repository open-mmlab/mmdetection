import time

from mmcv.runner import EpochBasedRunner
from mmcv.runner import RUNNERS
from mmcv.runner.utils import get_host_info

from .utils import check_nncf_is_enabled


@RUNNERS.register_module()
class AccuracyAwareRunner(EpochBasedRunner):
    """
    An mmdet training runner to be used with NNCF-based accuracy-aware training.
    Inherited from the standard EpochBasedRunner with the overridden "run" method.
    This runner does not use the "workflow" and "max_epochs" parameters that are
    used by the EpochBasedRunner since the training is controlled by NNCF's
    AdaptiveCompressionTrainingLoop that does the scheduling of the compression-aware
    training loop using the parameters specified in the "accuracy_aware_training".
    """

    def __init__(self, *args, target_metric_name, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_metric_name = target_metric_name

    def run(self, data_loaders, *args, compression_ctrl=None,
            nncf_config=None, configure_optimizers_fn=None):

        check_nncf_is_enabled()
        from nncf.common.accuracy_aware_training import create_accuracy_aware_training_loop
        assert isinstance(data_loaders, list)

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.warning('Note that the workflow and max_epochs parameters '
                            'are not used in NNCF-based accuracy-aware training')
        # taking only the first data loader for NNCF training
        self.train_data_loader = data_loaders[0]
        # Maximum possible number of iterations
        self._max_iters = self._max_epochs * len(self.train_data_loader)

        self.call_hook('before_run')

        acc_aware_training_loop = create_accuracy_aware_training_loop(nncf_config,
                                                                      compression_ctrl)
        model = acc_aware_training_loop.run(self.model,
                                            train_epoch_fn=self.train_fn,
                                            validate_fn=self.validation_fn,
                                            configure_optimizers_fn=configure_optimizers_fn)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
        return model

    def train_fn(self, *args, **kwargs):
        """
        Train the model for a single epoch.
        This method is used in NNCF-based accuracy-aware training.
        """
        self.train(self.train_data_loader)

    def validation_fn(self, *args, **kwargs):
        """
        Return the target metric value on the validation dataset.
        Evaluation is assumed to be already done at this point since EvalHook was called.
        This method is used in NNCF-based accuracy-aware training.
        """
        # Get metric from runner's attributes that set in EvalHook.evaluate() function
        metric = getattr(self, self.target_metric_name, None)
        if not metric:
            raise RuntimeError(f'Could not find the {self.target_metric_name} key')
        return metric
