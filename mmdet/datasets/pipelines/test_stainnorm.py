from tqdm import tqdm
import os
import cv2 as cv
from stainNorm_Vahadane import Normalizer
import stain_utils as ut

file_name = '/mnt/disk_share/wanglichao/mmdetection_data/VOC_Mitosis/VOC2007_ICPR_r480_cls2/JPEGImages/ICPR_A00_00_0.png'
saveFile = '/mnt/disk_share/wanglichao/mmdetection_data/VOC_Mitosis/'

method = Normalizer()
image = ut.read_image(file_name)
method.fit(ut.read_image('./data/MUS-AIHCGQLR.tif'))
normalized = method.transform(image)
normalized = cv.cvtColor(normalized, cv.COLOR_RGB2BGR)
cv.imwrite(os.path.join(saveFile, "sn_vahadane.jpg"), normalized)