import pytest

from hyped import augmenters


@pytest.mark.parametrize("name", augmenters.__all__)
def test_lazy_imports(name):
    """Test that all specified lazy imports are valid."""
    try:
        getattr(augmenters, name)
    except (ImportError, AttributeError):
        pytest.fail(f"Cannot import {name} from lazy augmenters module.")
