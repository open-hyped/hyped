import pytest

from hyped.io import writers


@pytest.mark.parametrize("name", writers.__all__)
def test_lazy_imports(name):
    """Test that all specified lazy imports are valid."""
    try:
        getattr(writers, name)
    except (ImportError, AttributeError):
        pytest.fail(f"Cannot import {name} from lazy io.writers module.")
