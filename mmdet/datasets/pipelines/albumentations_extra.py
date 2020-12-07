import random

import albumentations


class RandomRotate90and270(albumentations.RandomRotate90):

    def get_params(self):
        return {"factor": random.choice([0, 1, 3])}


ALBUMENTATIONS_EXTRA = {
    'RandomRotate90and270': RandomRotate90and270
}
