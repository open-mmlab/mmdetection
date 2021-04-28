import os

from .task_types import MMDetectionTaskType
from tasks.mmdetection_tasks.detection import MMDetectionParameters


class ConfigMappings(object):
    def __init__(self, task_type=MMDetectionTaskType.OBJECTDETECTION):
        """
        Class containing the mappings between values of NOUS configurable parameters and MMdetection configuration
        files.

        :param task_type: MMDetectionTaskType that specifies the type of task
        """

        # This path points to the mmdetection_tasks directory, i.e. the parent directory containing the config,
        # datasets, schedules, etc.
        base_task_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        default_data_pipeline_path = os.path.join(base_task_path, 'datasets/default_data_pipeline.py')

        if task_type == MMDetectionTaskType.OBJECTDETECTION:
            model_directory = os.path.join(base_task_path, 'thirdparty/models/detection/')
            self.model_file_map = {
                'faster-rcnn': dict(filename=os.path.join(model_directory, 'faster_rcnn_r50_fpn.py'),
                                    data_pipeline=default_data_pipeline_path,
                                    gradient_clipping=None),
                'cascade-rcnn': dict(filename=os.path.join(model_directory, 'cascade_rcnn_r101_fpn.py'),
                                     data_pipeline=default_data_pipeline_path,
                                     gradient_clipping=dict(max_norm=35, norm_type=2)),
                'retinanet': dict(filename=os.path.join(model_directory, 'retinanet_r101_fpn.py'),
                                  data_pipeline=default_data_pipeline_path,
                                  gradient_clipping=dict(max_norm=35, norm_type=2)),
                'yolov3': dict(filename=os.path.join(model_directory, 'yolov3.py'),
                               data_pipeline='../../datasets/yolo_data_pipeline.py',
                               gradient_clipping=dict(max_norm=100, norm_type=2))}
            self.configurable_parameter_type = MMDetectionParameters

        elif task_type == MMDetectionTaskType.INSTANCESEGMENTATION:
            model_directory = os.path.join(base_task_path, 'thirdparty/models/instance_segmentation/')
            self.model_file_map = {'mask-rcnn': dict(filename=os.path.join(model_directory, 'mask_rcnn_r101_fpn.py'),
                                                     data_pipeline=default_data_pipeline_path,
                                                     gradient_clipping=None)}

        # Base dir for the learning rate schedule config files
        schedule_dir = os.path.join(base_task_path, 'schedules')

        self.learning_rate_schedule_map = {
            'fixed': dict(filename=os.path.join(schedule_dir, 'schedule_fixed.py'), name='Fixed'),
            'step': dict(filename=os.path.join(schedule_dir, 'schedule_step.py'), name='Step-wise annealing'),
            'cyclic': dict(filename=os.path.join(schedule_dir, 'schedule_cyclic.py'), name='Cyclic cosine annealing'),
            'exp': dict(filename=os.path.join(schedule_dir, 'schedule_exp.py'), name='Exponential annealing')}

        # Base dir for the runtime config files
        runtime_dir = os.path.join(base_task_path, 'runtimes')

        self.runtime_map = {
            'default': dict(filename=os.path.join(runtime_dir, 'default_runtime.py'), name='Default')
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

