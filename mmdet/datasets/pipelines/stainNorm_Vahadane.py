# -*- coding: utf-8 -*-
"""
Stain normalization inspired by method of:

A. Vahadane et al., ‘Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images’, IEEE Transactions on Medical Imaging, vol. 35, no. 8, pp. 1962–1971, Aug. 2016.

Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

from __future__ import division

import spams
import numpy as np
import stain_utils as ut
# import cv2 as cv


def get_stain_matrix(I, threshold=0.8, lamda=0.1):
    """
    Get 2x3 stain matrix. First row H and second row E
    :param I:
    :param threshold:
    :param lamda:
    :return:
    """
    mask = ut.notwhite_mask(I, thresh=threshold).reshape((-1,))
    OD = ut.RGB_to_OD(I).reshape((-1, 3))
    OD = OD[mask]
    dictionary = spams.trainDL(OD.T, K=2, lambda1=lamda, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False).T
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    dictionary = ut.normalize_rows(dictionary)
    return dictionary


###

class Normalizer(object):
    """
    A stain normalization object
    """

    def __init__(self):
        self.stain_matrix_target = None

    def fit(self, target):
        target = ut.standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(target)

    def target_stains(self):
        return ut.OD_to_RGB(self.stain_matrix_target)

    def transform(self, I):
        I = ut.standardize_brightness(I)
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        return (255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target).reshape(I.shape))).astype(
            np.uint8)

    def hematoxylin(self, I):
        I = ut.standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        H = source_concentrations[:, 0].reshape(h, w)
        H = np.exp(-1 * H)
        return H



if __name__ == "__main__":
    from tqdm import tqdm
    import os
    import cv2 as cv

    file_name = '/mnt/disk_share/wanglichao/mmdetection_data/VOC_Mitosis/VOC2007_ICPR_r480_cls2/JPEGImages/ICPR_A00_00_0.png'
    saveFile = '/mnt/disk_share/wanglichao/mmdetection_data/VOC_Mitosis/'

    method = Normalizer()
    image = ut.read_image(file_name)
    method.fit(ut.read_image('./data/MUS-AIHCGQLR.tif'))
    normalized = method.transform(image)
    normalized = cv.cvtColor(normalized, cv.COLOR_RGB2BGR)
    cv.imwrite(os.path.join(saveFile, "sn_vahadane.jpg"), normalized)

    # image = ut.read_image('/mnt/disk_share/wanglichao/mmdetection_data/VOCdevkit/VOC2007_v1.2/JPEGImages_without_SCN/01_110.jpg')
    # method = Normalizer()
    # method.fit(ut.read_image('./data/MUS-AIHCGQLR.tif'))
    # normalized = method.transform(image)
    # normalized = cv.cvtColor(normalized, cv.COLOR_RGB2BGR)
    # cv.imwrite('01_110_N.jpg', normalized)
    # images_path = "/mnt/disk_share/wanglichao/mmdetection_data/VOCdevkit/VOC2007_v1.2/JPEGImages_without_SCN/"
    # saveFile = "/mnt/disk_share/wanglichao/mmdetection_data/VOCdevkit/VOC2007_v1.2/JPEGImages_with_Vahadane/"
    # method = Normalizer()
    # from tqdm import tqdm
    # import os
    # for file in tqdm(os.listdir(images_path)):
    #     file_name = os.path.join(images_path, file)
    #     image = ut.read_image(file_name)
    #     method.fit(ut.read_image('./data/MUS-AIHCGQLR.tif'))
    #     normalized = method.transform(image)
    #     normalized = cv.cvtColor(normalized, cv.COLOR_RGB2BGR)
    #     cv.imwrite(os.path.join(saveFile, file), normalized)
