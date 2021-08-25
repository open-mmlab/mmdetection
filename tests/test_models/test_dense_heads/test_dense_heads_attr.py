# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from terminaltables import AsciiTable

from mmdet.models import dense_heads
from mmdet.models.dense_heads import *  # noqa: F401,F403


def test_dense_heads_test_attr():
    """Tests inference methods such as simple_test and aug_test."""
    # make list of dense heads
    exceptions = ['FeatureAdaption']  # module used in head
    all_dense_heads = [m for m in dense_heads.__all__ if m not in exceptions]

    # search attributes
    check_attributes = [
        'simple_test', 'aug_test', 'simple_test_bboxes', 'simple_test_rpn',
        'aug_test_rpn'
    ]
    table_header = ['head name'] + check_attributes
    table_data = [table_header]
    not_found = {k: [] for k in check_attributes}
    for target_head_name in all_dense_heads:
        target_head = globals()[target_head_name]
        target_head_attributes = dir(target_head)
        check_results = [target_head_name]
        for check_attribute in check_attributes:
            found = check_attribute in target_head_attributes
            check_results.append(found)
            if not found:
                not_found[check_attribute].append(target_head_name)
        table_data.append(check_results)
    table = AsciiTable(table_data)
    print()
    print(table.table)

    # NOTE: this test just checks attributes.
    # simple_test of RPN heads will not work now.
    assert len(not_found['simple_test']) == 0, \
        f'simple_test not found in {not_found["simple_test"]}'
    if len(not_found['aug_test']) != 0:
        warnings.warn(f'aug_test not found in {not_found["aug_test"]}. '
                      'Please implement it or raise NotImplementedError.')
