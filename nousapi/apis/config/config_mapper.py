import os.path as osp

from .task_types import MMDetectionTaskType
from ..detection import MMDetectionParameters, list_available_models


class ConfigMappings(object):
    def __init__(self, task_type=MMDetectionTaskType.OBJECTDETECTION):
        """
        Class containing the mappings between values of NOUS configurable parameters and MMdetection configuration
        files.

        :param task_type: MMDetectionTaskType that specifies the type of task
        """

        base_task_path = osp.join(osp.abspath(osp.dirname(__file__)), '..', '..')

        self.model_file_map = {}
        if task_type == MMDetectionTaskType.OBJECTDETECTION:
            model_directory = osp.join(base_task_path, '..', 'configs', 'ote', 'custom-object-detection')
            available_models = list_available_models(model_directory)
            for model in available_models:
                self.model_file_map[model['name']] = dict(
                    filename=osp.join(model['dir'], 'model.py'),
                    data_pipeline=osp.join(model['dir'], 'nous_data_pipeline.py'),
                    gradient_clipping=None
                )
            self.configurable_parameter_type = MMDetectionParameters
        else:
            raise NotImplementedError()

        # Base dir for the learning rate schedule config files
        schedule_dir = osp.join(base_task_path, 'configs', 'schedules')
        self.learning_rate_schedule_map = {
            'fixed': dict(filename=osp.join(schedule_dir, 'schedule_fixed.py'), name='Fixed'),
            'step': dict(filename=osp.join(schedule_dir, 'schedule_step.py'), name='Step-wise annealing'),
            'cyclic': dict(filename=osp.join(schedule_dir, 'schedule_cyclic.py'), name='Cyclic cosine annealing'),
            'exp': dict(filename=osp.join(schedule_dir, 'schedule_exp.py'), name='Exponential annealing')}

        self.runtime_map = {
            'default': dict(filename=osp.join(base_task_path, 'configs', 'default_runtime.py'), name='Default')
        }

    def get_model_file(self, model_name: str) -> str:
        """ Returns the path to a file containing the configuration corresponding to a certain model name """
        return self.model_file_map[model_name]['filename']

    def get_schedule_file(self, schedule_name: str) -> str:
        """Returns the path to a file containing the configuration corresponding to a certain learning rate schedule"""
        return self.learning_rate_schedule_map[schedule_name]['filename']

    def get_schedule_friendly_name(self, schedule_name: str) -> str:
        """Returns the user friendly name of a certain learning rate schedule"""
        return self.learning_rate_schedule_map[schedule_name]['name']

    def get_data_pipeline_file(self, model_name: str) -> str:
        """ Returns the path to a file containing the data pipeline config corresponding to a certain model name """
        return self.model_file_map[model_name]['data_pipeline']

    def get_gradient_clipping(self, model_name: str) -> str:
        """ Returns whether or not to use gradient clipping for the model defined by model_name """
        return self.model_file_map[model_name]['gradient_clipping']

    def get_runtime_file(self, runtime_name: str) -> str:
        return self.runtime_map[runtime_name]['filename']

