try:
    from pytest_cov.embed import cleanup_on_sigterm
except ImportError:
    pass
else:
    cleanup_on_sigterm()

import pytest

import hyped.common._worker
from hyped.common._worker import manager


@pytest.fixture(autouse=True)
def _reset_globals_after_test():
    yield
    # reset multiprocessing worker info
    hyped.common._worker._worker_info = None
    # reset global manager
    if manager._is_instantiated():
        manager._instance.shutdown()
        manager._instance = None
        assert not manager._is_instantiated()
