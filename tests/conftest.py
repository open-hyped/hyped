try:
    from pytest_cov.embed import cleanup_on_sigterm
except ImportError:
    pass
else:
    cleanup_on_sigterm()

# set multiprocessing start method compatible with pytest
# see: https://github.com/pytest-dev/pytest/issues/11174
import multiprocessing as mp

mp.set_start_method("spawn", force=True)
