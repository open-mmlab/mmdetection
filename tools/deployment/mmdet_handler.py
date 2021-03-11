import base64
import os

import mmcv
import torch
from ts.torch_handler.base_handler import BaseHandler

from mmdet.apis import inference_detector, init_detector


class MMDetHandler(BaseHandler):
    threshold = 0.5

    def initialize(self, context):
        """Init MMDetection model."""

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
        if isinstance(data, tuple):
            bbox_result, segm_result = data
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = data, None

        # Format output following the example ObjectDetectionHandler format
        output = []
        for class_index, class_result in enumerate(bbox_result):
            class_name = self.model.CLASSES[class_index]
            for bbox in class_result:
                bbox_coords = bbox[:-1]
                score = bbox[-1]
                if score > self.threshold:
                    output.append({class_name: bbox_coords, 'score': score})

        return output
