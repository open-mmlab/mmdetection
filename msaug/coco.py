from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from copy import deepcopy

@DATASETS.register_module()
class CocoAugDataset(CocoDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 pre_pipeline=None,
                 multiscale_mode_student=None,
                 ratio_hr_lr_student=None,
                 min_lr_student=None,
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
        
        if (pre_pipeline is None) or (self.test_mode):
            self.pre_train_pipeline = None
        else:
            self.pre_train_pipeline = Compose(pre_pipeline)
        
        if not self.test_mode:
            pipeline_teacher = []
            pipeline_student = []
            for pipe_teacher in pipeline:
                if pipe_teacher['type'] == 'Resize':
                    # Check Single or Multi-scale
                    if type(pipe_teacher['img_scale']) == tuple:
                        single = True
                    else:
                        if len(pipe_teacher['img_scale']) == 1:
                            single = True
                        else:
                            single = False
                            
                    if single:
                        # single-scale -> resizing with ratio (r)
                        pipe_student = deepcopy(pipe_teacher)
                        pipe_student.update({'type': 'Resize_Student', 'multiscale_mode': multiscale_mode_student, 'ratio_hr_lr': ratio_hr_lr_student, 'ratio_range': (min_lr_student, 1.0)})
                    else:
                        # multi-scale -> no resizing for student input
                        pipe_student = deepcopy(pipe_teacher)
                        pipe_teacher = dict(type='Resize', img_scale=(1333, 800), keep_ratio=True)
                        
                    pipeline_teacher.append(pipe_teacher)
                    pipeline_student.append(pipe_student)
                                    
                else:
                    pipeline_teacher.append(pipe_teacher)
                    pipeline_student.append(pipe_teacher)

            self.pipeline = Compose(pipeline_teacher)
            self.pipeline_student = Compose(pipeline_student)


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
        
        if self.pre_train_pipeline is not None:
            self.pre_train_pipeline(results)
            
        results_original, results_augment = deepcopy(results), deepcopy(results)
        return self.pipeline(results_original), self.pipeline_student(results_augment)
     
        
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