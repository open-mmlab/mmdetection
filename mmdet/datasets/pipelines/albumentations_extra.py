import random

import albumentations
import numpy as np

class RandomRotate90and270(albumentations.RandomRotate90):

    def get_params(self):
        return {"factor": random.choice([0, 1, 3])}


class RandomRotate(albumentations.Rotate):

    def __call__(self, **results):
        res = super().__call__(**results)
        res['texts'] = res['texts'].tolist()
        for i, bbox in enumerate(res['bboxes']):
            xmin, ymin, xmax, ymax, _ = bbox
            if xmin < 0 or ymin < 0 or xmax > 1.0 or ymax > 1.0:
                res['texts'][i] = np.array([])
        res['texts'] = np.array(res['texts'])
        return res

ALBUMENTATIONS_EXTRA = {
    'RandomRotate90and270': RandomRotate90and270,
    'RandomRotate': RandomRotate,
}
