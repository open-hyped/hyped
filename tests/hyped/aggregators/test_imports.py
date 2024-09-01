import pytest

from hyped import aggregators


@pytest.mark.parametrize("name", aggregators.__all__)
def test_lazy_imports(name):
    """Test that all specified lazy imports are valid."""
    try:
        getattr(aggregators, name)
    except (ImportError, AttributeError):
        pytest.fail(f"Cannot import {name} from lazy aggregators module.")
