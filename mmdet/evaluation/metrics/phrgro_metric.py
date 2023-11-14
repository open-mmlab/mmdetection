# Copyright (c) OpenMMLab. All rights reserved
import sys
import numpy as np
import os.path as osp
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import mmengine

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet.registry import METRICS
from ..functional import eval_recalls

def _recalls(all_ious, proposal_nums, thrs):

    img_num = all_ious.shape[0]
    total_gt_num = sum([ious.shape[0] for ious in all_ious])

    _ious = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_ious = np.zeros(0)
        for i in range(img_num):
            ious = all_ious[i][:, :proposal_num].copy()
            gt_ious = np.zeros((ious.shape[0]))
            if ious.size == 0:
                tmp_ious = np.hstack((tmp_ious, gt_ious))
                continue
            for j in range(ious.shape[0]):
                gt_max_overlaps = ious.argmax(axis=1)
                max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                ious[gt_idx, :] = -1
                ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        _ious[k, :] = tmp_ious

    _ious = np.fliplr(np.sort(_ious, axis=1))
    recalls = np.zeros((proposal_nums.size, thrs.size))
    for i, thr in enumerate(thrs):
        recalls[:, i] = (_ious >= thr).sum(axis=1) / float(total_gt_num)

    return recalls

def _adjust_labels(arr):
    for i in range(0, max(arr)):
        if i not in arr:
            arr = [x - 1 if x > i else x for x in arr]
            return _adjust_labels(arr)
    return arr

class RecallTracker:
    """ Utility class to track recall@k for various k, split by categories"""

    def __init__(self, topk: Sequence[int]):
        """
        Parameters:
           - topk : tuple of ints corresponding to the recalls being tracked (eg, recall@1, recall@10, ...)
        """

        self.total_byk_bycat: Dict[int, Dict[str, int]] = {k: defaultdict(int) for k in topk}
        self.positives_byk_bycat: Dict[int, Dict[str, int]] = {k: defaultdict(int) for k in topk}

    def add_positive(self, k: int, category: str):
        """Log a positive hit @k for given category"""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f"{k} is not a valid recall threshold")
        self.total_byk_bycat[k][category] += 1
        self.positives_byk_bycat[k][category] += 1

    def add_negative(self, k: int, category: str):
        """Log a negative hit @k for given category"""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f"{k} is not a valid recall threshold")
        self.total_byk_bycat[k][category] += 1

    def report(self) -> Dict[int, Dict[str, float]]:
        """Return a condensed report of the results as a dict of dict.
        report[k][cat] is the recall@k for the given category
        """
        report: Dict[int, Dict[str, float]] = {}
        for k in self.total_byk_bycat:
            assert k in self.positives_byk_bycat
            report[k] = {
                cat: self.positives_byk_bycat[k][cat] / self.total_byk_bycat[k][cat] for cat in self.total_byk_bycat[k]
            }
        return report

@METRICS.register_module()
class PhrGroMetric(BaseMetric):
    """Phrase Grounding Metric.
    """
    def __init__(self,
                 flickr_path: str,
                 subset: str = 'test',
                 topk: Sequence[int] = (1, 5, 10),
                 iou_thrs: float = 0.5,
                 merge_boxes: bool = False,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.iou_thrs = iou_thrs
        self.topk = topk
        
        # text to sentence_id
        self.imgid2sentenceid: Dict[str, str] = {}
        anns = mmengine.load(ann_file, file_format='json')
        for img in anns['images']:
            img_id = img['id']
            sentence_id = img['sentence_id']
            self.imgid2sentenceid[img_id] = sentence_id

        # load image ids of the current subset
        with open(osp.join(flickr_path, f"{subset}.txt")) as f:
            self.img_ids = [line.strip() for line in f]
        
        # load the box annotations for all images
        self.imgid2boxes: Dict[str, Dict[str, List[List[int]]]] = {}
        for img_id in self.img_ids:
            anno_info = self.get_annotations(osp.join(flickr_path, "Annotations", f"{img_id}.xml"))["boxes"]
            if merge_boxes:
                merged = {}
                for phrase_id, boxes in anno_info.items():
                    merged[phrase_id] = self.merge_boxes(boxes)
                anno_info = merged
            self.imgid2boxes[img_id] = anno_info

        # load the sentences annotations
        self.imgid2sentences: Dict[str, List[List[Optional[Dict]]]] = {}
        self.all_ids: List[str] = []
        tot_phrases = 0
        for img_id in self.img_ids:
            sentence_info = self.get_sentence_data(osp.join(flickr_path, "Sentences", f"{img_id}.txt"))
            self.imgid2sentences[img_id] = [None for _ in range(len(sentence_info))]

            # deal with the phrases that don't have boxes
            for sent_id, sentence in enumerate(sentence_info):
                phrases = [phrase for phrase in sentence["phrases"] if phrase["phrase_id"] in self.imgid2boxes[img_id]]
                if len(phrases) > 0:
                    self.imgid2sentences[img_id][sent_id] = phrases
                tot_phrases += len(phrases)
            
            self.all_ids += [
                f"{img_id}_{k}" for k in range(len(sentence_info)) if self.imgid2sentences[img_id][k] is not None
            ]

    def get_annotations(self, filename) -> Dict[str, Union[int, List[str], Dict[str, List[List[int]]]]]:
        """
        parses the xml files in the Flickr30K Entities dataset

        input:
          filema,e - full file path to the annotation file to parse
        
        output:
          dictionary with the following fields:
            scene - list of identifiers which were annotated as 
                    pertaining the whole scene
            nobox - list of identifiers which were annotated as 
                    not being visible in the image
            boxes - a dictionary where the fields are identifiers
                    and the values are its list of boxes in the
                    [xmin ymin xmax ymax] format
            height - int representing the height of the image
            width - int representing the width of the image
            depth - int representing the depth of the image
        """
        import xml.etree.ElementTree as ET
        tree = ET.parse(filename)
        root = tree.getroot()
        
        anno_info: Dict[str, Union[int, List[str], Dict[str, List[List[int]]]]] = {}
        all_boxes: Dict[str, List[List[int]]] = {}
        all_noboxes: List[str] = []
        all_scenes: List[str] = []

        size_container = root.findall("size")[0]
        for size_element in size_container:
            assert size_element.text
            anno_info[size_element.tag] = int(size_element.text)

        for object_container in root.findall("object"):
            for names in object_container.findall("name"):
                box_id = names.text
                assert box_id
                box_container = object_container.findall("bndbox")
                if len(box_container) > 0:
                    if box_id not in all_boxes:
                        all_boxes[box_id] = []
                    xmin = int(box_container[0].findall("xmin")[0].text)
                    ymin = int(box_container[0].findall("ymin")[0].text)
                    xmax = int(box_container[0].findall("xmax")[0].text)
                    ymax = int(box_container[0].findall("ymax")[0].text)
                    all_boxes[box_id].append([xmin, ymin, xmax, ymax])
                else:
                    nobndbox = int(object_container.findall("nobndbox")[0].text)
                    if nobndbox > 0:
                        all_noboxes.append(box_id)
                    
                    scene = int(object_container.findall("scene")[0].text)
                    if scene > 0:
                        all_scenes.append(box_id)
        
        anno_info["boxes"] = all_boxes
        anno_info["nobox"] = all_noboxes
        anno_info["scene"] = all_scenes

        return anno_info

    def get_sentence_data(self, filename) -> List[Dict[str, Any]]:
        """
        Parses a sentence file from the Flickr30K Entities dataset

        input:
          filename - full file path to the sentence file to parse
        
        output:
          a list of 
        """
        with open(filename, 'r') as f:
            sentences = f.read().split('\n')
        
        annotations = []
        for sentence in sentences:
            if not sentence:
                continue

            first_word = []
            phrases = []
            phrase_id = []
            phrase_type = []
            words = []
            current_phrase = []
            add_to_phrase = False
            for token in sentence.split():
                if add_to_phrase:
                    if token[-1] == "]":
                        add_to_phrase = False
                        token = token[:-1]
                        current_phrase.append(token)
                        phrases.append(" ".join(current_phrase))
                        current_phrase = []
                    else:
                        current_phrase.append(token)

                    words.append(token)
                else:
                    if token[0] == "[":
                        add_to_phrase = True
                        first_word.append(len(words))
                        parts = token.split('/')
                        phrase_id.append(parts[1][3:])
                        phrase_type.append(parts[2:])
                    else:
                        words.append(token)
            
            sentence_data = {"sentence": " ".join(words), "phrases": []}
            for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
                sentence_data["phrases"].append(
                    {"first_word_index": index, "phrase": phrase, "phrase_id": p_id, "phrase_type": p_type}
                )

            annotations.append(sentence_data)

        return annotations

    def merge_boxes(self, boxes: List[List[int]]) -> List[List[int]]:
        """
        Return the boxes corresponding to the smallest enclosing box containing all the provided boxes
        The boxes are expected in [x1, y1, x2, y2] format
        """
        if len(boxes) == 1:
            return boxes
        
        np_boxes = np.asarray(boxes)

        return [[np.boxes[:, 0].min(), np_boxes[:, 1].min(), np_boxes[:, 2].max(), np_boxes[:, 3].max()]]


    def _box_area(self, boxes: np.array) -> np.array:
        assert boxes.ndim == 2 and boxes.shape[-1] == 4
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


    def _box_inter_union(self, boxes1: np.array, boxes2: np.array) -> Tuple[np.array, np.array]:
        area1 = self._box_area(boxes1)
        area2 = self._box_area(boxes2)

        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2]) # [N,M,2]
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:]) # [N,M,2]

        wh = (rb - lt).clip(min=0) # [N,M,2]

        inter = wh[:, :, 0] * wh[:, :, 1] # [N,M]
        union = area1[:, None] + area2 - inter

        return inter, union

    def box_iou(self, boxes1: np.array, boxes2: np.array) -> np.array:
        inter, union = self._box_inter_union(boxes1, boxes2)
        iou = inter / union
        return iou

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """

        evaluated_ids = set()

        for data_sample in data_samples:
            cid = data_sample['img_id']
            img_id = data_sample['img_path'].split('/')[-1].split('.')[0]
            # print(img_id)
            sentence_id = self.imgid2sentenceid[cid]
            pred = data_sample['pred_instances']
            # print("{}_{}".format(img_id, sentence_id))
            cur_id = "{}_{}".format(img_id, sentence_id)
            if cur_id in evaluated_ids:
                print(
                    "Warning, multiple predictions found for sentence {} in image {}".format(sentence_id, img_id)
                )
                continue
            
            # Skip the sentences with no valid phrase
            if cur_id not in self.all_ids:
                if len(pred['bboxes']) != 0:
                    print(
                        "Warning, in image {} we were not expecting predictions for sentence {}. ignoring them.".format(img_id, sentence_id)
                    )
                continue

            evaluated_ids.add(cur_id)
            
            pred_boxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            '''
            print('cid', cid)
            print('b', pred_boxes)
            print(len(pred_boxes))
            print('s', pred_scores)
            print(len(pred_scores))
            print('l', pred_labels)
            print(len(pred_labels))
            '''
            
            phrases = self.imgid2sentences[str(img_id)][int(sentence_id)]
        
            gt_boxes = []
            pred_id = 0

            for pred_id in range(min(pred_labels), max(pred_labels) + 1):
                cur_boxes = []
                for k, label in enumerate(pred_labels):
                    if label == pred_id:
                        tmp = pred_boxes[k].tolist()
                        tmp.append(pred_scores[k])
                        cur_boxes.append(tmp)
                if len(cur_boxes) == 0:
                    continue
                try:
                    phrase = phrases[pred_id]
                    target_boxes = self.imgid2boxes[str(img_id)][phrase['phrase_id']]
                except:
                    target_boxes = None
                    continue
                cur_boxes.sort(key=lambda x: x[4], reverse=True)
                self.results.append((cur_boxes, target_boxes))

            """
            for phrase in phrases:
                '''
                if int(phrase['phrase_id']) < rec:
                    print(rec)
                    print(phrase['phrase_id'])
                    print(phrases)
                    sys.exit()
                rec = int(phrase['phrase_id'])
                '''

                
                
                for k, label in enumerate(pred_labels):
                    if label == pred_id:
                        tmp = pred_boxes[k].tolist()
                        tmp.append(pred_scores[k])
                        # print(a)
                        
                        # import sys
                        # sys.exit()
                        cur_boxes.append(tmp)
                
                pred_id += 1

                target_boxes = self.imgid2boxes[str(img_id)][phrase['phrase_id']]
                
                
                
                # print(type(target_boxes))
                # print(type(cur_boxes))
                # print(cur_boxes)
                # import sys
                # sys.exit()

                if len(cur_boxes) == 0:
                    cur_boxes.append([0,0,0,0,0])
                    print(len(phrases))
                    atp = set(pred_labels)
                    print(len(atp))
                    if len(atp) != len(phrases):
                        print(phrases)
                        print(pred_labels)
                        sys.exit()
                    '''
                    print(target_boxes)
                    import sys
                    
                    '''
            
                self.results.append((cur_boxes, target_boxes))
                """

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        preds, gts = zip(*results)

        gt_boxes = [np.array(gt) if gt is not None else None for gt in gts]
        pred_boxes = [np.array(pred) for pred in preds]

        # all_ious = np.array(results)

        # eval_gts = [np.array(gt) for gt in gts]
        
        eval_results = OrderedDict()

        # recalls = _recalls(all_ious, self.topk, self.iou_thrs)
        recalls = eval_recalls(gt_boxes, pred_boxes, self.topk, self.iou_thrs)

        print(recalls)

        return eval_results
