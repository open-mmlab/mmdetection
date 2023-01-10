from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from copy import deepcopy

@DATASETS.register_module()
class CocoContDataset(CocoDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 multiscale_mode_student=None,
                 ratio_range_student=None,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 seg_suffix='.png',
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):

        super(CocoContDataset, self).__init__(ann_file,
                 pipeline,
                 classes,
                 data_root,
                 img_prefix,
                 seg_prefix,
                 seg_suffix,
                 proposal_file,
                 test_mode,
                 filter_empty_gt,
                 file_client_args)
        
        if not self.test_mode:
            pipeline_multiscale = []
            for pipe in pipeline:
                if pipe['type'] == 'Resize':
                    pipe.update({'type': 'Resize_Student', 'multiscale_mode': multiscale_mode_student, 'ratio_range': ratio_range_student})
                pipeline_multiscale.append(pipe)
            
            self.pipeline_multiscale = Compose(pipeline_multiscale)


    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        
        results_original, results_augment = deepcopy(results), deepcopy(results)
        return self.pipeline(results_original), self.pipeline_multiscale(results_augment)
     
        
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        
        while True:
            data_ori, data_aug = self.prepare_train_img(idx)
            if data_ori is None:
                idx = self._rand_another(idx)
                continue
            
            # Duplicate data
            return data_ori, data_aug