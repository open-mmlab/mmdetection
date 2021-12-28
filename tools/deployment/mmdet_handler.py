# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os

import mmcv
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
        for image_index, image_result in enumerate(data):
            output.append([])
            if isinstance(image_result, tuple):
                bbox_result, segm_result = image_result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = image_result, None

            for class_index, class_result in enumerate(bbox_result):
                class_name = self.model.CLASSES[class_index]
                for bbox in class_result:
                    bbox_coords = bbox[:-1].tolist()
                    score = float(bbox[-1])
                    if score >= self.threshold:
                        output[image_index].append({
                            'class_name': class_name,
                            'bbox': bbox_coords,
                            'score': score
                        })

        return output
