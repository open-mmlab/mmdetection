import os

import cv2

from custom.convert.utils import read_kitti_annot

an_path = "/home/chrissikek/repos/data/brummer/val/label_2"
img_path = "/home/chrissikek/repos/data/brummer/val/image_2"
out_path = "/home/chrissikek/repos/data/brummer/val/ann_vis"

def draw_bb(image, bb, text):
    x, y, x2, y2 = [int(p) for p in bb]
    image = cv2.rectangle(image, (x, y, x2-x, y2-y), (36, 255, 12), 1)
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


os.makedirs(out_path, exist_ok=True)
for annot in os.listdir(an_path):
    img_name = annot[:-4] + ".jpg"
    fimg_path = os.path.join(img_path, img_name)
    img = cv2.imread(fimg_path)
    annot_path = os.path.join(an_path, annot)
    a = read_kitti_annot(annot_path)
    for bb, label in a:
        draw_bb(img, bb, label)
    write_path = os.path.join(out_path, img_name)
    cv2.imwrite(write_path, img)




