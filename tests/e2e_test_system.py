# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
import copy
import socket
import datetime
import functools
import pytest
import traceback
import getpass
import time



def _generate_e2e_pytest_decorators():
    try:
        from e2e.markers.mark_meta import MarkMeta
    except ImportError:
        def _e2e_pytest_api(func):
            return func
        def _e2e_pytest_performance(func):
            return func
        return _e2e_pytest_api, _e2e_pytest_performance

    class Requirements:
        # Dummy requirement
        REQ_DUMMY = "Dummy requirement"

    class OTEComponent(MarkMeta):
        OTE = "ote"

    def _e2e_pytest_api(func):
        @pytest.mark.components(OTEComponent.OTE)
        @pytest.mark.priority_medium
        @pytest.mark.reqids(Requirements.REQ_DUMMY)
        @pytest.mark.api_other
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    def _e2e_pytest_performance(func):
        @pytest.mark.components(OTEComponent.OTE)
        @pytest.mark.priority_medium
        @pytest.mark.reqids(Requirements.REQ_DUMMY)
        @pytest.mark.api_performance
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    return _e2e_pytest_api, _e2e_pytest_performance

def _create_class_DataCollector():
    try:
        from e2e.collection_system.systems import TinySystem
        return TinySystem
    except ImportError:
        class _dummy_DataCollector: #should have the same interface as TinySystem
            def __init__(self, *args, **kwargs):
                pass

            def flush(self):
                pass
            def register_collector(self, *args, **kwargs):
                pass
            def register_exporter(self, *args, **kwargs):
                pass
            def log_final_metric(self, *args, **kwargs):
                pass
            def log_internal_metric(self, *args, **kwargs):
                pass
            def update_metadata(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, typerr, value, tback):
                if typerr is not None:
                    traceback.format_tb(tback)
                    raise typerr(value)
                return True

        return _dummy_DataCollector

e2e_pytest_api, e2e_pytest_performance = _generate_e2e_pytest_decorators()
DataCollector = _create_class_DataCollector()
