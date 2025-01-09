try:
    from pytest_cov.embed import cleanup_on_sigterm
except ImportError:
    pass
else:
    cleanup_on_sigterm()

import pytest

import hyped.common._worker


@pytest.fixture(autouse=True)
def _reset_globals_after_test():
    yield
    # reset multiprocessing worker info
    hyped.common._worker._worker_info = None
