# Copyright (c) Github URL
# Copied from
# https://github.com/youtubevos/cocoapi/blob/master/PythonAPI/pycocotools/ytvos.py
__author__ = 'ychfan'
# Interface for accessing the YouTubeVIS dataset.

# The following API functions are defined:
#  YTVIS       - YTVIS api class that loads YouTubeVIS annotation file
#  and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  loadRes    - Load algorithm results and create API for accessing them.

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import copy
import itertools
import json
import sys
import time
from collections import defaultdict

import numpy as np
from pycocotools import mask as maskUtils

PYTHON_VERSION = sys.version_info[0]


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class YTVIS:

    def __init__(self, annotation_file=None):
        """Constructor of Microsoft COCO helper class for reading and
        visualizing annotations.

        :param annotation_file (str | dict): location of annotation file or
            dict results.
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.vids = dict(), dict(), dict(
        ), dict()
        self.vidToAnns, self.catToVids = defaultdict(list), defaultdict(list)
        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            if type(annotation_file) == str:
                dataset = json.load(open(annotation_file, 'r'))
            else:
                dataset = annotation_file
            assert type(
                dataset
            ) == dict, 'annotation file format {} not supported'.format(
                type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, vids = {}, {}, {}
        vidToAnns, catToVids = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                vidToAnns[ann['video_id']].append(ann)
                anns[ann['id']] = ann

        if 'videos' in self.dataset:
            for vid in self.dataset['videos']:
                vids[vid['id']] = vid

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToVids[ann['category_id']].append(ann['video_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.vidToAnns = vidToAnns
        self.catToVids = catToVids
        self.vids = vids
        self.cats = cats

    def getAnnIds(self, vidIds=[], catIds=[], areaRng=[], iscrowd=None):
        """Get ann ids that satisfy given filter conditions. default skips that
        filter.

        :param vidIds  (int array)     : get anns for given vids
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range
               iscrowd (boolean)       : get anns for given crowd label
        :return: ids (int array)       : integer array of ann ids
        """
        vidIds = vidIds if _isArrayLike(vidIds) else [vidIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(vidIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(vidIds) == 0:
                lists = [
                    self.vidToAnns[vidId] for vidId in vidIds
                    if vidId in self.vidToAnns
                ]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [
                ann for ann in anns if ann['category_id'] in catIds
            ]
            anns = anns if len(areaRng) == 0 else [
                ann for ann in anns if ann['avg_area'] > areaRng[0]
                and ann['avg_area'] < areaRng[1]
            ]
        if iscrowd is not None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """filtering parameters. default skips that filter.

        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [
                cat for cat in cats if cat['name'] in catNms
            ]
            cats = cats if len(supNms) == 0 else [
                cat for cat in cats if cat['supercategory'] in supNms
            ]
            cats = cats if len(catIds) == 0 else [
                cat for cat in cats if cat['id'] in catIds
            ]
        ids = [cat['id'] for cat in cats]
        return ids

    def getVidIds(self, vidIds=[], catIds=[]):
        """Get vid ids that satisfy given filter conditions.

        :param vidIds (int array) : get vids for given ids
        :param catIds (int array) : get vids with all given cats
        :return: ids (int array)  : integer array of vid ids
        """
        vidIds = vidIds if _isArrayLike(vidIds) else [vidIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(vidIds) == len(catIds) == 0:
            ids = self.vids.keys()
        else:
            ids = set(vidIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToVids[catId])
                else:
                    ids &= set(self.catToVids[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """Load anns with the specified ids.

        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """Load cats with the specified ids.

        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadVids(self, ids=[]):
        """Load anns with the specified ids.

        :param ids (int array)       : integer ids specifying vid
        :return: vids (object array) : loaded vid objects
        """
        if _isArrayLike(ids):
            return [self.vids[id] for id in ids]
        elif type(ids) == int:
            return [self.vids[ids]]

    def loadRes(self, resFile):
        """Load result file and return a result api object.

        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = YTVIS()
        res.dataset['videos'] = [img for img in self.dataset['videos']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or (PYTHON_VERSION == 2
                                    and type(resFile) == str):
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsVidIds = [ann['video_id'] for ann in anns]
        assert set(annsVidIds) == (set(annsVidIds) & set(self.getVidIds())), \
               'Results do not correspond to current coco set'
        if 'segmentations' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(
                self.dataset['categories'])
            for id, ann in enumerate(anns):
                ann['areas'] = []
                if 'bboxes' not in ann:
                    ann['bboxes'] = []
                for seg in ann['segmentations']:
                    # now only support compressed RLE format
                    # as segmentation results
                    if seg:
                        ann['areas'].append(maskUtils.area(seg))
                        if len(ann['bboxes']) < len(ann['areas']):
                            ann['bboxes'].append(maskUtils.toBbox(seg))
                    else:
                        ann['areas'].append(None)
                        if len(ann['bboxes']) < len(ann['areas']):
                            ann['bboxes'].append(None)
                ann['id'] = id + 1
                l_ori = [a for a in ann['areas'] if a]
                if len(l_ori) == 0:
                    ann['avg_area'] = 0
                else:
                    ann['avg_area'] = np.array(l_ori).mean()
                ann['iscrowd'] = 0
        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def annToRLE(self, ann, frameId):
        """Convert annotation which can be polygons, uncompressed RLE to RLE.

        :return: binary mask (numpy 2D array)
        """
        t = self.vids[ann['video_id']]
        h, w = t['height'], t['width']
        segm = ann['segmentations'][frameId]
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = segm
        return rle

    def annToMask(self, ann, frameId):
        """Convert annotation which can be polygons, uncompressed RLE, or RLE
        to binary mask.

        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, frameId)
        m = maskUtils.decode(rle)
        return m
