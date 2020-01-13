"""DEEPSCORES

Provides access to the QualitAI database with a COCO-like interface. The
only changes made compared to the coco.py file are the class labels.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    November 23, 2019
"""
from .coco import *


@DATASETS.register_module
class QualitaiDataset(CocoDataset):

    CLASSES = ('bad')
