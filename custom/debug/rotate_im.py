import os

import mmcv
import cv2

def rotate_image(img, angle):
    img_rotated = mmcv.imrotate(img, angle, auto_bound=True)
    return img_rotated


path = "data/fst/random_select_/random_select"
path2 = "data/fst/random_select_/random_select_rotated"
for im in os.listdir(path):
    image = cv2.imread(os.path.join(path, im))
    if image.shape[1] > image.shape[0]:
        image = rotate_image(image, 90)
    cv2.imwrite(os.path.join(path2, im), image)

