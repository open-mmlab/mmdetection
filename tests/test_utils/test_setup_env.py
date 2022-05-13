import datetime
import sys
from unittest import TestCase

from mmengine import DefaultScope

from mmdet.utils import register_all_modules


class TestSetupEnv(TestCase):

    def test_register_all_modules(self):
        from mmdet.registry import DATASETS

        # not init default scope
        sys.modules.pop('mmdet.datasets', None)
        sys.modules.pop('mmdet.datasets.coco', None)
        DATASETS._module_dict.pop('CocoDataset', None)
        self.assertFalse('CocoDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=False)
        self.assertTrue('CocoDataset' in DATASETS.module_dict)

        # init default scope
        sys.modules.pop('mmdet.datasets')
        sys.modules.pop('mmdet.datasets.coco')
        DATASETS._module_dict.pop('CocoDataset', None)
        self.assertFalse('CocoDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=True)
        self.assertTrue('CocoDataset' in DATASETS.module_dict)
        self.assertEqual(DefaultScope.get_current_instance().scope_name,
                         'mmdet')

        # init default scope when another scope is init
        name = f'test-{datetime.datetime.now()}'
        DefaultScope.get_instance(name, scope_name='test')
        with self.assertWarnsRegex(
                Warning, 'The current default scope "test" is not "mmdet"'):
            register_all_modules(init_default_scope=True)
