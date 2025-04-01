import torch
from mmengine.dataset import RepeatDataset
from mmengine.hooks import Hook
from mmengine.runner import Runner
from tqdm import tqdm

from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import HOOKS
from mmdet.structures.bbox import bbox2roi


@HOOKS.register_module()
class PseudoLabelClusteringHook(Hook):

    def __init__(self, calculate_pseudo_labels_from_epoch: int = 0) -> None:
        self.calculate_pseudo_labels_from_epoch = calculate_pseudo_labels_from_epoch

    def before_train_epoch(self, runner: Runner) -> None:
        runner.model.roi_head.epoch = runner.epoch
        if runner.epoch >= self.calculate_pseudo_labels_from_epoch:
            with torch.no_grad():
                ann_ids = []
                for i, data_batch in enumerate(tqdm(runner.train_dataloader)):
                    data = runner.model.data_preprocessor(data_batch, True)
                    bboxes = [data_sample.gt_instances.bboxes for data_sample in data['data_samples']]
                    fts = runner.model.extract_feat(data['inputs'])
                    rois = bbox2roi(bboxes)
                    gt_ann_ids = [data_sample.gt_instances.ann_ids for data_sample in data['data_samples']]
                    ann_ids.extend(gt_ann_ids)
                    runner.model.roi_head.accumulate_pseudo_labels(fts, rois)

                labels = runner.model.roi_head.calculate_pseudo_labels().numpy()
                ann_ids = torch.cat(ann_ids).cpu().numpy()

                pseudo_classes = {ann_id: label for ann_id, label in zip(ann_ids, labels)}
                runner.train_dataloader.dataset.cat_pseudo_label_mapping = pseudo_classes
                # runner.train_dataloader.dataset._fully_initialized = False
                # runner.train_dataloader.dataset.full_init()

                # Changing _fully_initialized to False and calling full_init() again doesn't work
                # load data information
                runner.train_dataloader.dataset.data_list = runner.train_dataloader.dataset.load_data_list()
                # get proposals from file
                if runner.train_dataloader.dataset.proposal_file is not None:
                    runner.train_dataloader.dataset.load_proposals()
                # filter illegal data, such as data that has no annotations.
                runner.train_dataloader.dataset.data_list = runner.train_dataloader.dataset.filter_data()

                # Get subset data according to indices.
                if runner.train_dataloader.dataset._indices is not None:
                    runner.train_dataloader.dataset.data_list = \
                        runner.train_dataloader.dataset._get_unserialized_subset(runner.train_dataloader.dataset._indices)

                # serialize data_list
                if runner.train_dataloader.dataset.serialize_data:
                    runner.train_dataloader.dataset.data_bytes, runner.train_dataloader.dataset.data_address = runner.train_dataloader.dataset._serialize_data()

                # dataset = runner.train_dataloader.dataset.coco.dataset
                # for ann_id, label in zip(ann_ids, labels):
                #     if type(runner.train_dataloader.dataset) == RepeatDataset:
                #         runner.train_dataloader.dataset.dataset.coco.anns[ann_id]['pseudo_class'] = label
                #     else:
                #         runner.train_dataloader.dataset.coco.anns[ann_id]['pseudo_class'] = label
                #     new_coco = COCO()
                #     new_coco.dataset = dataset
                #     new_coco.createIndex()
                #     new_coco.img_ann_map = new_coco.imgToAnns
                #     new_coco.cat_img_map = new_coco.catToImgs
                #     runner.train_dataloader.dataset.coco = new_coco
