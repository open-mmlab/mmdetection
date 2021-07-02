# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os.path as osp

from ..detection import MMDetectionParameters

class ConfigMappings(object):
    def __init__(self):
        """
        Class containing the mappings between values of OTE configurable parameters and MMdetection configuration
        files.
        """

        base_task_path = osp.join(osp.abspath(osp.dirname(__file__)), '..', '..')

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

    def get_schedule_file(self, schedule_name: str) -> str:
        """Returns the path to a file containing the configuration corresponding to a certain learning rate schedule"""
        return self.learning_rate_schedule_map[schedule_name]['filename']

    def get_schedule_friendly_name(self, schedule_name: str) -> str:
        """Returns the user friendly name of a certain learning rate schedule"""
        return self.learning_rate_schedule_map.get(schedule_name, {'name': 'custom'})['name']

    def get_runtime_file(self, runtime_name: str) -> str:
        return self.runtime_map[runtime_name]['filename']

