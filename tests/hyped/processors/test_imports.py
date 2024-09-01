import pytest

from hyped import processors


@pytest.mark.parametrize("name", processors.__all__)
def test_lazy_imports(name):
    """Test that all specified lazy imports are valid."""
    try:
        getattr(processors, name)
    except (ImportError, AttributeError):
        pytest.fail(f"Cannot import {name} from lazy processors module.")
