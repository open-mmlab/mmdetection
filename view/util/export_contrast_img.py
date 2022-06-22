import os
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.image import imread

root_dir = r''
origin_dir = r''
lka_dir = r''

origin_total_dir = osp.join(root_dir,origin_dir)
lka_total_dir = osp.join(root_dir,lka_dir)

origin_imgs = os.listdir(origin_total_dir)
lka_imgs = os.listdir(lka_total_dir)

# for ori_img,lka_img in origin_imgs,lka_imgs:
#     plt