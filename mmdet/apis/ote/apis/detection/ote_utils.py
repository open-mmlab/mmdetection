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

import importlib
import os
import subprocess
import tempfile
import yaml
from sc_sdk.entities.label import Color, Label, distinct_colors
from sc_sdk.entities.label_schema import (LabelGroup, LabelGroupType,
                                          LabelSchema)


def generate_label_schema(label_names):
    label_domain = "detection"
    colors = distinct_colors(len(label_names)) if len(label_names) > 0 else []
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
