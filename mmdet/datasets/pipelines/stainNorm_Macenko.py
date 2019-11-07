# -*- coding: utf-8 -*-
"""
Stain normalization based on the method of:

M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.

Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

from __future__ import division
import numpy as np
import stain_utils as ut


def get_stain_matrix(I, pic_name=None, beta=0.15, alpha=1):
    """
    Get stain matrix (2x3)
    :param I:
    :param beta:
    :param alpha:
    :return:
    """
    OD = ut.RGB_to_OD(I).reshape((-1, 3))
    OD = (OD[(OD > beta).any(axis=1), :])
    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
    V = V[:, [2, 1]]
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1
    That = np.dot(OD, V)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])
    return ut.normalize_rows(HE)


###

class Normalizer(object):
    """
    A stain normalization object
    """

    def __init__(self):
        self.stain_matrix_target = None
        self.target_concentrations = None

    def fit(self, target):
        target = ut.standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(target)
        self.target_concentrations = ut.get_concentrations(target, self.stain_matrix_target)

    def target_stains(self):
        return ut.OD_to_RGB(self.stain_matrix_target)

    def transform(self, I):
        I = ut.standardize_brightness(I)
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= (maxC_target / maxC_source)
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
    cv.imwrite(os.path.join(saveFile, "sn_macenko.jpg"), normalized)

    # images_path = "/mnt/disk_share/wanglichao/mmdetection_data/VOCdevkit/VOC2007_v1.2/JPEGImages_without_SCN/"
    # saveFile = "/mnt/disk_share/wanglichao/mmdetection_data/VOCdevkit/VOC2007_v1.2/JPEGImages_with_Macenko/"
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
