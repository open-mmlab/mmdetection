# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) https://github.com/mcordts/cityscapesScripts
# A wrapper of `cityscapesscripts` which supports loading groundtruth
# image from `backend_args`.
import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

import mmcv
import numpy as np
from mmengine.fileio import get

try:
    import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as CSEval  # noqa: E501
    from cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling import \
        CArgs  # noqa: E501
    from cityscapesscripts.evaluation.instance import Instance
    from cityscapesscripts.helpers.csHelpers import (id2label, labels,
                                                     writeDict2JSON)
    HAS_CITYSCAPESAPI = True
except ImportError:
    CArgs = object
    HAS_CITYSCAPESAPI = False


def evaluateImgLists(prediction_list: list,
                     groundtruth_list: list,
                     args: CArgs,
                     backend_args: Optional[dict] = None,
                     dump_matches: bool = False) -> dict:
    """A wrapper of obj:``cityscapesscripts.evaluation.

    evalInstanceLevelSemanticLabeling.evaluateImgLists``. Support loading
    groundtruth image from file backend.
    Args:
        prediction_list (list): A list of prediction txt file.
        groundtruth_list (list): A list of groundtruth image file.
        args (CArgs): A global object setting in
            obj:``cityscapesscripts.evaluation.
            evalInstanceLevelSemanticLabeling``
        backend_args (dict, optional): Arguments to instantiate the
            preifx of uri corresponding backend. Defaults to None.
        dump_matches (bool): whether dump matches.json. Defaults to False.
    Returns:
        dict: The computed metric.
    """
    if not HAS_CITYSCAPESAPI:
        raise RuntimeError('Failed to import `cityscapesscripts`.'
                           'Please try to install official '
                           'cityscapesscripts by '
                           '"pip install cityscapesscripts"')
    # determine labels of interest
    CSEval.setInstanceLabels(args)
    # get dictionary of all ground truth instances
    gt_instances = getGtInstances(
        groundtruth_list, args, backend_args=backend_args)
    # match predictions and ground truth
    matches = matchGtWithPreds(prediction_list, groundtruth_list, gt_instances,
                               args, backend_args)
    if dump_matches:
        CSEval.writeDict2JSON(matches, 'matches.json')
    # evaluate matches
    apScores = CSEval.evaluateMatches(matches, args)
    # averages
    avgDict = CSEval.computeAverages(apScores, args)
    # result dict
    resDict = CSEval.prepareJSONDataForResults(avgDict, apScores, args)
    if args.JSONOutput:
        # create output folder if necessary
        path = os.path.dirname(args.exportFile)
        CSEval.ensurePath(path)
        # Write APs to JSON
        CSEval.writeDict2JSON(resDict, args.exportFile)

    CSEval.printResults(avgDict, args)

    return resDict


def matchGtWithPreds(prediction_list: list,
                     groundtruth_list: list,
                     gt_instances: dict,
                     args: CArgs,
                     backend_args=None):
    """A wrapper of obj:``cityscapesscripts.evaluation.

    evalInstanceLevelSemanticLabeling.matchGtWithPreds``. Support loading
    groundtruth image from file backend.
    Args:
        prediction_list (list): A list of prediction txt file.
        groundtruth_list (list): A list of groundtruth image file.
        gt_instances (dict): Groundtruth dict.
        args (CArgs): A global object setting in
            obj:``cityscapesscripts.evaluation.
            evalInstanceLevelSemanticLabeling``
        backend_args (dict, optional): Arguments to instantiate the
            preifx of uri corresponding backend. Defaults to None.
    Returns:
        dict: The processed prediction and groundtruth result.
    """
    if not HAS_CITYSCAPESAPI:
        raise RuntimeError('Failed to import `cityscapesscripts`.'
                           'Please try to install official '
                           'cityscapesscripts by '
                           '"pip install cityscapesscripts"')
    matches: dict = dict()
    if not args.quiet:
        print(f'Matching {len(prediction_list)} pairs of images...')

    count = 0
    for (pred, gt) in zip(prediction_list, groundtruth_list):
        # Read input files
        gt_image = readGTImage(gt, backend_args)
        pred_info = readPredInfo(pred)
        # Get and filter ground truth instances
        unfiltered_instances = gt_instances[gt]
        cur_gt_instances_orig = CSEval.filterGtInstances(
            unfiltered_instances, args)

        # Try to assign all predictions
        (cur_gt_instances,
         cur_pred_instances) = CSEval.assignGt2Preds(cur_gt_instances_orig,
                                                     gt_image, pred_info, args)

        # append to global dict
        matches[gt] = {}
        matches[gt]['groundTruth'] = cur_gt_instances
        matches[gt]['prediction'] = cur_pred_instances

        count += 1
        if not args.quiet:
            print(f'\rImages Processed: {count}', end=' ')
            sys.stdout.flush()

    if not args.quiet:
        print('')

    return matches


def readGTImage(image_file: Union[str, Path],
                backend_args: Optional[dict] = None) -> np.ndarray:
    """Read an image from path.

    Same as obj:``cityscapesscripts.evaluation.
    evalInstanceLevelSemanticLabeling.readGTImage``, but support loading
    groundtruth image from file backend.
    Args:
        image_file (str or Path): Either a str or pathlib.Path.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
    Returns:
        np.ndarray: The groundtruth image.
    """
    img_bytes = get(image_file, backend_args=backend_args)
    img = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow')
    return img


def readPredInfo(prediction_file: str) -> dict:
    """A wrapper of obj:``cityscapesscripts.evaluation.

    evalInstanceLevelSemanticLabeling.readPredInfo``.
    Args:
        prediction_file (str): The prediction txt file.
    Returns:
        dict: The processed prediction results.
    """
    if not HAS_CITYSCAPESAPI:
        raise RuntimeError('Failed to import `cityscapesscripts`.'
                           'Please try to install official '
                           'cityscapesscripts by '
                           '"pip install cityscapesscripts"')
    printError = CSEval.printError

    predInfo = {}
    if (not os.path.isfile(prediction_file)):
        printError(f"Infofile '{prediction_file}' "
                   'for the predictions not found.')
    with open(prediction_file) as f:
        for line in f:
            splittedLine = line.split(' ')
            if len(splittedLine) != 3:
                printError('Invalid prediction file. Expected content: '
                           'relPathPrediction1 labelIDPrediction1 '
                           'confidencePrediction1')
            if os.path.isabs(splittedLine[0]):
                printError('Invalid prediction file. First entry in each '
                           'line must be a relative path.')

            filename = os.path.join(
                os.path.dirname(prediction_file), splittedLine[0])

            imageInfo = {}
            imageInfo['labelID'] = int(float(splittedLine[1]))
            imageInfo['conf'] = float(splittedLine[2])  # type: ignore
            predInfo[filename] = imageInfo

    return predInfo


def getGtInstances(groundtruth_list: list,
                   args: CArgs,
                   backend_args: Optional[dict] = None) -> dict:
    """A wrapper of obj:``cityscapesscripts.evaluation.

    evalInstanceLevelSemanticLabeling.getGtInstances``. Support loading
    groundtruth image from file backend.
    Args:
        groundtruth_list (list): A list of groundtruth image file.
        args (CArgs): A global object setting in
            obj:``cityscapesscripts.evaluation.
            evalInstanceLevelSemanticLabeling``
        backend_args (dict, optional): Arguments to instantiate the
            preifx of uri corresponding backend. Defaults to None.
    Returns:
        dict: The computed metric.
    """
    if not HAS_CITYSCAPESAPI:
        raise RuntimeError('Failed to import `cityscapesscripts`.'
                           'Please try to install official '
                           'cityscapesscripts by '
                           '"pip install cityscapesscripts"')
    # if there is a global statistics json, then load it
    if (os.path.isfile(args.gtInstancesFile)):
        if not args.quiet:
            print('Loading ground truth instances from JSON.')
        with open(args.gtInstancesFile) as json_file:
            gt_instances = json.load(json_file)
    # otherwise create it
    else:
        if (not args.quiet):
            print('Creating ground truth instances from png files.')
        gt_instances = instances2dict(
            groundtruth_list, args, backend_args=backend_args)
        writeDict2JSON(gt_instances, args.gtInstancesFile)

    return gt_instances


def instances2dict(image_list: list,
                   args: CArgs,
                   backend_args: Optional[dict] = None) -> dict:
    """A wrapper of obj:``cityscapesscripts.evaluation.

    evalInstanceLevelSemanticLabeling.instances2dict``. Support loading
    groundtruth image from file backend.
    Args:
        image_list (list): A list of image file.
        args (CArgs): A global object setting in
            obj:``cityscapesscripts.evaluation.
            evalInstanceLevelSemanticLabeling``
        backend_args (dict, optional): Arguments to instantiate the
            preifx of uri corresponding backend. Defaults to None.
    Returns:
        dict: The processed groundtruth results.
    """
    if not HAS_CITYSCAPESAPI:
        raise RuntimeError('Failed to import `cityscapesscripts`.'
                           'Please try to install official '
                           'cityscapesscripts by '
                           '"pip install cityscapesscripts"')
    imgCount = 0
    instanceDict = {}

    if not isinstance(image_list, list):
        image_list = [image_list]

    if not args.quiet:
        print(f'Processing {len(image_list)} images...')

    for image_name in image_list:
        # Load image
        img_bytes = get(image_name, backend_args=backend_args)
        imgNp = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow')

        # Initialize label categories
        instances: dict = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp):
            instanceObj = Instance(imgNp, instanceId)

            instances[id2label[instanceObj.labelID].name].append(
                instanceObj.toDict())

        instanceDict[image_name] = instances
        imgCount += 1

        if not args.quiet:
            print(f'\rImages Processed: {imgCount}', end=' ')
            sys.stdout.flush()

    return instanceDict
