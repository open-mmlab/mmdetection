import cv2
import numpy as np
import matplotlib.pyplot as plt
import mmcv


def draw_bbs(results):
    image, bbs = results["img"], results["gt_bboxes"]
    image = image.copy()
    for bb in bbs:
        cv2.rectangle(image, (int(bb[0]), int(bb[1]), int(bb[2] - bb[0]), int(bb[3] - bb[1])), (0, 255, 0), 2)
    return image


class Rot90:
    def __init__(self):
        pass

    def __call__(self, results):
        angle = np.random.choice([0., 90., 180., 270.])
        results["img"] = self.rotate_image(results["img"], angle)
        self.rotate_anots(results, angle)
        if angle % 180 != 0:
            results["img_shape"] = results["img_shape"][1], results["img_shape"][0], *results["img_shape"][2:]
        return results

    def rotate_image(self, img, angle):
        img_rotated = mmcv.imrotate(img, angle, auto_bound=True)
        return img_rotated

    def rotate_anots(self, results, angle):
        for key in results.get('bbox_fields', []):
            results[key] = [self.rot90bb(bbox, angle, (results["img_shape"][1], results["img_shape"][0])) for bbox in results[key]]



    def rot90coords(self, coords, angle, shape):
        # xyxy
        print(shape)
        print(coords)
        for i in range(0, len(coords), 2):
            coord_x = coords[i]
            coord_y = coords[i + 1]
            shape_x, shape_y = shape
            for _ in range(int(angle // 90)):
                coord_y_new = coord_x
                coord_x_new = shape_y - coord_y
                shape_x, shape_y = shape_y, shape_x
                coord_x, coord_y = coord_x_new, coord_y_new
            coords[i] = coord_x
            coords[i + 1] = coord_y
        return coords

    def rot90bb(self, bb, angle, shape):
        points = self.rot90coords(bb, angle, shape)
        x = (points[0], points[2])
        y = (points[1], points[3])
        return [min(x), min(y), max(x), max(y)]

#img = np.random.randint(low=0, high=255, size=(100,200,3), dtype=np.uint8)
img = cv2.imread("data/fst/random_select_/random_select/img_2022_10_04_11_08_33t674979.jpg")
bbs = [[350, 800, 450, 830]]
results = {"img": img, "gt_bboxes":bbs, "img_shape": img.shape, "bbox_fields": ["gt_bboxes"]}
cv2.imwrite("data/dump/x1.png", draw_bbs(results))
t = Rot90()
results= t(results)
cv2.imwrite("data/dump/x2.png", draw_bbs(results))