try:
    import e2e.fixtures

    from e2e.conftest_utils import * # noqa
    from e2e.conftest_utils import pytest_addoption as _e2e_pytest_addoption # noqa
    from e2e import config # noqa
    from e2e.utils import get_plugins_from_packages
    pytest_plugins = get_plugins_from_packages([e2e])
except ImportError:
    _e2e_pytest_addoption = None
    pass

def pytest_addoption(parser):
    if _e2e_pytest_addoption:
        _e2e_pytest_addoption(parser)
    parser.addoption('--dataset-definitions', action='store', default=None,
                     help='Path to the dataset_definitions.yml file for tests that require datasets.')
