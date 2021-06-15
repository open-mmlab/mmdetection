"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import json
import os
from copy import copy


def load_nncf_config(path):
    assert path.endswith('.json'), (
            f'Only json files are allowed as optimisation configs, provided {path}')
    with open(path) as f_src:
        nncf_config  = json.load(f_src)
    return nncf_config


def compose_nncf_config(nncf_config, enabled_options):
    optimisation_parts = nncf_config

    if 'order_of_parts' in optimisation_parts:
        # The result of applying the changes from optimisation parts
        # may depend on the order of applying the changes
        # (e.g. if for nncf_quantization it is sufficient to have `total_epochs=2`,
        #  but for sparsity it is required `total_epochs=50`)
        # So, user can define `order_of_parts` in the optimisation_config
        # to specify the order of applying the parts.
        order_of_parts = optimisation_parts['order_of_parts']
        assert isinstance(order_of_parts, list), \
            'The field "order_of_parts" in optimisation config should be a list'

        for part in enabled_options:
            assert part in order_of_parts, (
                    f'The part {part} is selected, but it is absent in order_of_parts={order_of_parts},'
                    f' see the optimisation config file {config_path}')

        optimisation_parts_to_choose = [part for part in order_of_parts if part in enabled_options]

    assert 'base' in optimisation_parts, 'Error: the optimisation config does not contain the "base" part'
    nncf_config_part = optimisation_parts['base']

    for part in optimisation_parts_to_choose:
        assert part in optimisation_parts, (
                f'Error: the optimisation config does not contain the part "{part}", '
                f'whereas it was selected; see the optimisation config file "{config_path}"')
        optimisation_part_dict = optimisation_parts[part]
        try:
            nncf_config_part = merge_dicts_and_lists_b_into_a(nncf_config_part, optimisation_part_dict)
        except AssertionError as cur_error:
            err_descr = (f'Error during merging the parts of nncf configs from file "{config_path}":\n'
                f'the current part={part}, '
                f'the order of merging parts into base is {optimisation_parts_to_choose}.\n'
                f'The error is:\n{cur_error}')
            raise RuntimeError(err_descr) from None

    return nncf_config_part


def merge_dicts_and_lists_b_into_a(a, b):
    return _merge_dicts_and_lists_b_into_a(a, b, "")


def _merge_dicts_and_lists_b_into_a(a, b, cur_key=None):
    """The function is inspired by mmcf.Config._merge_a_into_b,
    but it
    * works with usual dicts and lists and derived types
    * supports merging of lists (by concatenating the lists)
    * makes recursive merging for dict + dict case
    * overwrites when merging scalar into scalar
    Note that we merge b into a (whereas Config makes merge a into b),
    since otherwise the order of list merging is counter-intuitive.
    """
    def _err_str(_a, _b, _key):
        if _key is None:
            _key_str = 'of whole structures'
        else:
            _key_str = f'during merging for key=`{_key}`'
        return (f'Error in merging parts of config: different types {_key_str},'
                f' type(a) = {type(_a)},'
                f' type(b) = {type(_b)}')

    assert isinstance(a, (dict, list)), f'Can merge only dicts and lists, whereas type(a)={type(a)}'
    assert isinstance(b, (dict, list)), _err_str(a, b, cur_key)
    assert isinstance(a, list) == isinstance(b, list), _err_str(a, b, cur_key)
    if isinstance(a, list):
        # the main diff w.r.t. mmcf.Config -- merging of lists
        return a + b

    a = copy(a)
    for k in b.keys():
        if k not in a:
            a[k] = copy(b[k])
            continue
        new_cur_key = cur_key + '.' + k if cur_key else k
        if isinstance(a[k], (dict, list)):
            a[k] = _merge_dicts_and_lists_b_into_a(a[k], b[k], new_cur_key)
            continue

        assert not isinstance(b[k], (dict, list)), _err_str(a[k], b[k], new_cur_key)

        # suppose here that a[k] and b[k] are scalars, just overwrite
        a[k] = b[k]
    return a
