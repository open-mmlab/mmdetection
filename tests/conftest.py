import e2e.fixtures

from e2e.conftest_utils import * # noqa
from e2e import config # noqa
from e2e.utils import get_plugins_from_packages
pytest_plugins = get_plugins_from_packages([e2e])

