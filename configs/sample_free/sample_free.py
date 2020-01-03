import argparse
import json
import os
import re
from math import ceil
from pathlib import Path

import torch
from mmcv import Config
from tqdm import tqdm

from mmdet.apis import init_detector
from mmdet.core.anchor import anchor_inside_flags
from mmdet.core.bbox import build_assigner
from mmdet.datasets import build_dataloader, build_dataset


class SampleFree(object):
    """
    sample-free include Bias Initialization, Guided Loss, Class-Adaptive
    Threshold, to keep the origin mmdet code structure, we compute prior
    and set score_thr before training.

    for one-stage, we compute prior based on the assign result, and simply
    set this prior for score_thr.
    for two-stage, we compute prior based on the assign result for rpn stage,
    while for roi-stage, there is no need to compute the prior,
    and the score_thr is simply set 0.001

    more detail information can refer to the code
    https://github.com/ChenJoya/sampling-free

    Args:
        cfg (str): config file path
        detector_type (str): 'one-stage' or 'two-stage'
        datset_name (str): name of dataset, eg: voc, coco

    """

    def __init__(self, cfg_file, detector_type, datset_name, device='cuda'):
        assert detector_type in ['one-stage', 'two-stage']
        self.cfg_file = cfg_file
        self.cfg = Config.fromfile(cfg_file)
        self.detector_type = detector_type
        self.dataset_name = datset_name
        self.device = device
        self.prior = self.compute_prior(self.cfg)

    def compute_prior(self, cfg):
        if self.detector_type == 'one-stage':
            head = 'bbox_head'
            assign_cfg = cfg.train_cfg.assigner
            allowed_border = cfg.train_cfg.allowed_border
        else:
            head = 'rpn_head'
            assign_cfg = cfg.train_cfg.rpn.assigner
            allowed_border = cfg.train_cfg.rpn.allowed_border

        # set bigger to speed up
        imgs_per_gpu = 16
        workers_per_gpu = 16

        # dataloader
        if cfg.data.train.type == 'RepeatDataset':
            cfg.data.train.times = 1
        dataset = build_dataset(cfg.data.train)
        dataloader = build_dataloader(dataset, imgs_per_gpu, workers_per_gpu)

        # model
        # Note: before build model, we need get the model config
        # without sample-free
        head_dict = cfg.model.get(head)
        head_dict['type'] = head_dict['head']
        head_dict.pop('model_key')
        head_dict.pop('init_prior_file')
        head_dict.pop('head')
        head_dict.pop('loss_cls_scale')

        model = init_detector(cfg)
        strides = getattr(model, head).anchor_strides
        num_class = getattr(model, head).num_classes - 1

        # assigner
        assigner = build_assigner(assign_cfg)

        num_fg, num_all = 0, 0
        for img_data in tqdm(dataloader):
            img_metas, img, gt_bboxes = img_data['img_meta'].data[
                0], img_data['img'].data[0], img_data['gt_bboxes'].data[0]
            h, w = img.shape[-2:]
            featmap_sizes = [(int(ceil(h / stride)), int(ceil(w / stride)))
                             for stride in strides]

            anchors, valid_flags = getattr(model, head).get_anchors(
                featmap_sizes, img_metas, device=self.device)
            anchors = [torch.cat(anchors_per_image, 0)
                       for anchors_per_image in anchors]
            valid_flags = [torch.cat(valid_flags_per_image, 0)
                           for valid_flags_per_image in valid_flags]

            for anchor, valid_flag, gt_bbox, img_meta in zip(anchors,
                                                             valid_flags,
                                                             gt_bboxes,
                                                             img_metas):
                inside_flag = anchor_inside_flags(
                    anchor, valid_flag, img_meta['img_shape'][:2],
                    allowed_border)
                anchor = anchor[inside_flag, :]
                gt_bbox = gt_bbox.to(self.device)

                assign_result = assigner.assign(anchor, gt_bbox)
                num_fg_per_image = (assign_result.gt_inds > 0).sum()
                num_bg_per_image = (assign_result.gt_inds == 0).sum()
                num_fg += num_fg_per_image
                num_all += (num_fg_per_image + num_bg_per_image)

        ratio = num_fg.float() / num_all.float()
        prior = ratio / num_class

        return float(prior.cpu())

    def request_model_info(self, cfg):
        detector = cfg.model.type
        backbone = cfg.model.backbone
        neck = cfg.model.neck
        model = "{}_{}{}_{}_{}".format(
            detector, backbone.type, backbone.depth,
            neck.type, self.dataset_name.upper())

        return model

    def save_prior(self, filename):
        model = self.request_model_info(self.cfg)
        self.model_key = model

        if os.path.exists(filename):
            with open(filename, 'r') as f:
                model_prior_dict = json.load(f)
                model_prior_dict[model] = self.prior
        else:
            print("{} is not existed. Create it.".format(filename))
            model_prior_dict = {model: self.prior}

        with open(filename, 'w') as f:
            json.dump(model_prior_dict, f)
        print("Priors have saved to {}.".format(filename))

    def set_score_thr(self):
        """rewrite the score_thr and model key in config file
        """
        score_thr = 0.001 if self.detector_type == 'two-stage' else self.prior
        cfg_text = Path(self.cfg_file).read_text()
        origin_score_thr = re.findall(r'score_thr=(.*?),', cfg_text)[0]
        cfg_text = cfg_text.replace(
            'score_thr={}'.format(origin_score_thr),
            'score_thr={}'.format(score_thr))

        origin_model_key = re.findall(r'model_key=(.*?),', cfg_text)[0]
        cfg_text = cfg_text.replace(
            "model_key={}".format(origin_model_key),
            "model_key='{}'".format(self.model_key))
        Path(self.cfg_file).write_text(cfg_text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--detector_type', default='two-stage',
                        help='detector is one-stage or two-stage')
    parser.add_argument('--dataset', default='voc', help='dataset name')
    parser.add_argument('--save_path',
                        default='configs/sample_free/init_prior.json',
                        help='save path of computed prior')
    args = parser.parse_args()
    return args


# run this code before training
if __name__ == "__main__":
    args = parse_args()
    sample_free = SampleFree(args.config, args.detector_type, args.dataset)
    sample_free.save_prior(args.save_path)
    sample_free.set_score_thr()
