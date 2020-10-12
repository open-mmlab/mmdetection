# Copyright (C) 2020 Intel Corporation
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
#

import numpy as np
import torch
from mmcv.parallel import collate, scatter

from mmdet.datasets.pipelines import Compose
from mmdet.parallel.data_cpu import scatter_cpu
from .inference import LoadImage

def get_fake_input(cfg, orig_img_shape=(128, 128, 3), device='cuda'):
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    data = dict(img=np.zeros(orig_img_shape, dtype=np.uint8))
    data = test_pipeline(data)
    if device == torch.device('cpu'):
        data = scatter_cpu(collate([data], samples_per_gpu=1))[0]
    else:
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    return data
