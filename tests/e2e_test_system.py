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

from e2e.logger import get_logger

from e2e.collection_system.core import CollectionManager
from e2e.collection_system.extentions import LoggerExporter
from e2e.collection_system.extentions import MongoExporter
from e2e.collection_system.core import Collector


def _generate_e2e_pytest_decorator():
    try:
        from e2e.markers.mark_meta import MarkMeta
    except ImportError:
        def _e2e_pytest(func):
            return func
        return _e2e_pytest

    class Requirements:
        # Dummy requirement
        REQ_DUMMY = "Dummy requirement"

    class OTEComponent(MarkMeta):
        OTE = "ote"

    def _e2e_pytest(func):
        @pytest.mark.components(OTEComponent.OTE)
        @pytest.mark.priority_medium
        @pytest.mark.reqids(Requirements.REQ_DUMMY)
        @pytest.mark.api_other
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    return _e2e_pytest

e2e_pytest = _generate_e2e_pytest_decorator()

def select_configurable_parameters(json_configurable_parameters):
    selected = {}
    getv = lambda c, n: c[n]["value"]
    for section, container in json_configurable_parameters.items():
        for param, values in container.items():
            try:
                selected[f"{section}_{param}"] = getv(container, param)
            except TypeError: print("######## TypeError:", section, param)
            except KeyError: print("######## KeyError:", section, param)
    return selected


class CollsysManager:
    logger = get_logger("CollsysMgr")
    metacoll_name = "metadata"

    def __init__(self, name, setup):
        self.logger.info("init")
        self.collmanager = CollectionManager()
        self.manual_collector = Collector(name=name)
        self.collmanager.register_collector(self.manual_collector)
        self.meta_collector = Collector(name=self.metacoll_name)
        self.collmanager.register_collector(self.meta_collector)
        
        cs_logger = LoggerExporter("logger", accept="final")
        self.collmanager.register_exporter(cs_logger)
        self._set_mongo_exporter(setup)

    def flush(self):
        self.collmanager.flush()
    def register_collector(self, collector):
        self.collmanager.register_collector(collector)
    def register_exporter(self, exporter):
        self.collmanager.register_exporter(exporter)
    def log_final_metric(self, key, val):
        self.manual_collector.log_final_metric(key, val)
    def log_internal_metric(self, key, val, flush=False, timestamp=None, offset=None):
        self.manual_collector.log_internal_metric(key, val, flush, timestamp, offset)
    def update_metadata(self, key, value):
        self.meta_collector.log_final_metric(key, value)

    def __enter__(self):
        self.logger.info("start")
        self.collmanager.start()
        self.start_ts = time.time()
        return self

    def __exit__(self, typerr, value, tback):
        duration = time.time() - self.start_ts
        self.manual_collector.log_final_metric("mgr_duration", duration)
        
        self.logger.info("stop")
        self.collmanager.flush()
        self.collmanager.stop()

        if typerr is not None:
            traceback.format_tb(tback)        
            raise typerr(value)
        return True

    def _set_mongo_exporter(self, setup):
        database_url = os.environ.get("TT_DATABASE_URL")
        if database_url is None:
            self.logger.warning("DB not configured! skiped...")
            return

        metadata = self._make_metadata(setup)
        self.cs_mongoexp = MongoExporter(db_url=database_url, metadata=metadata, metacoll=self.metacoll_name)
        self.collmanager.register_exporter(self.cs_mongoexp)    

    def _make_metadata(self, setup):
        metadata = copy.deepcopy(setup)
        metadata["system_user"] = getpass.getuser()
        metadata["client_hostname"] = socket.gethostname()
        metadata["execution_date"] = datetime.datetime.now()
        metadata["build_name_seed"] = os.environ.get("BUILD_NAME_SEED", "no-env")
        metadata["environment_name"] = os.environ.get("TT_ENVIRONMENT_NAME", "no-env")
        metadata["mmdetection_branch"] = os.environ.get("MMDETECTION_BRANCH_EXECUTION", "no-env")
        metadata["test_type"] = os.environ.get("TT_TEST_TYPE", "no-env")        
        return metadata
