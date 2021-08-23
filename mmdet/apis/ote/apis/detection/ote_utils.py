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

import colorsys
import importlib
import os
import random
import subprocess
import tempfile
import yaml
from typing import Optional

import numpy as np
from ote_sdk.entities.train_parameters import UpdateProgressCallback
from ote_sdk.entities.label import Color
from sc_sdk.entities.label import Label
from ote_sdk.entities.label_schema import LabelGroup, LabelGroupType
from ote_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from sc_sdk.entities.label_schema import LabelSchema


class ColorPalette:
    def __init__(self, n, rng=None):
        assert n > 0

        if rng is None:
            rng = random.Random(0xACE)

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, n):
            colors_candidates = [(rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0))
                                 for _ in range(candidates_num)]
            min_distances = [self.min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [Color(*self.hsv2rgb(*hsv)) for hsv in hsv_colors]

    @staticmethod
    def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @classmethod
    def min_distance(cls, colors_set, color_candidate):
        distances = [cls.dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    @staticmethod
    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    def __getitem__(self, n):
        return self.palette[n % len(self.palette)]

    def __len__(self):
        return len(self.palette)


def generate_label_schema(label_names):
    label_domain = "detection"
    colors = ColorPalette(len(label_names)) if len(label_names) > 0 else []
    not_empty_labels = [Label(name=name, color=colors[i], domain=label_domain, id=i) for i, name in
                        enumerate(label_names)]
    emptylabel = Label(name=f"Empty label", color=Color(42, 43, 46),
                       is_empty=True, domain=label_domain, id=len(not_empty_labels))

    label_schema = LabelSchema()
    exclusive_group = LabelGroup(name="labels", labels=not_empty_labels, group_type=LabelGroupType.EXCLUSIVE)
    empty_group = LabelGroup(name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL)
    label_schema.add_group(exclusive_group)
    label_schema.add_group(empty_group, exclusive_with=[exclusive_group])
    return label_schema


def load_template(path):
    with open(path) as f:
        template = yaml.full_load(f)
    return template


def get_task_class(path):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def reload_hyper_parameters(model_template):
    """ This function copies template.yaml file and its configuration.yaml dependency to temporal folder.
        Then it re-loads hyper parameters from copied template.yaml file.
        This function should not be used in general case, it is assumed that
        the 'configuration.yaml' should be in the same folder as 'template.yaml' file.
    """

    template_file = model_template.model_template_path
    template_dir = os.path.dirname(template_file)
    temp_folder = tempfile.mkdtemp()
    conf_yaml = [dep.source for dep in model_template.dependencies if dep.destination == model_template.hyper_parameters.base_path][0]
    conf_yaml = os.path.join(template_dir, conf_yaml)
    subprocess.run(f'cp {conf_yaml} {temp_folder}', check=True, shell=True)
    subprocess.run(f'cp {template_file} {temp_folder}', check=True, shell=True)
    model_template.hyper_parameters.load_parameters(os.path.join(temp_folder, 'template.yaml'))
    assert model_template.hyper_parameters.data


class TrainingProgressCallback(TimeMonitorCallback):
    def __init__(self, update_progress_callback: Optional[UpdateProgressCallback] = None):
        super().__init__(0, 0, 0, 0, update_progress_callback=update_progress_callback)

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        self.update_progress_callback(self.get_progress())
