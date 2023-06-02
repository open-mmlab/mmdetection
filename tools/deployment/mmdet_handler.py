# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os

import mmcv
import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

from mmdet.apis import inference_detector, init_detector


class MMdetHandler(BaseHandler):
    threshold = 0.5

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')

        self.model = init_detector(self.config_file, checkpoint, self.device)
        self.initialized = True

    def preprocess(self, data):
        images = []

        for row in data:
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            images.append(image)

        return images

    def inference(self, data, *args, **kwargs):
        results = inference_detector(self.model, data)
        return results

    def postprocess(self, data):
        # Format output following the example ObjectDetectionHandler format
        output = []
        for data_sample in data:
            pred_instances = data_sample.pred_instances
            bboxes = pred_instances.bboxes.cpu().numpy().astype(
                np.float32).tolist()
            labels = pred_instances.labels.cpu().numpy().astype(
                np.int32).tolist()
            scores = pred_instances.scores.cpu().numpy().astype(
                np.float32).tolist()
            preds = []
            for idx in range(len(labels)):
                cls_score, bbox, cls_label = scores[idx], bboxes[idx], labels[
                    idx]
                if cls_score >= self.threshold:
                    class_name = self.model.dataset_meta['classes'][cls_label]
                    result = dict(
                        class_label=cls_label,
                        class_name=class_name,
                        bbox=bbox,
                        score=cls_score)
                    preds.append(result)
            output.append(preds)
        return output
