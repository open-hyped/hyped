from unittest.mock import MagicMock, patch

import pytest

from hyped.common.lazy_module import LazyModule


@pytest.fixture
def lazy_module():
    lazy_imports = {"foo": "module_foo", "bar": "module_bar"}
    module_file = "/path/to/module.py"
    module_spec = "spec"
    return LazyModule(
        "lazy_module",
        "A lazy-loaded module",
        module_file,
        module_spec,
        lazy_imports,
    )


@patch("importlib.import_module")
def test_lazy_import(mock_import_module, lazy_module):
    mock_module = MagicMock()
    mock_import_module.return_value = mock_module
    mock_module.foo = "foo_value"

    # Access the attribute to trigger the lazy loading
    foo_value = lazy_module.foo

    # Check that import_module was called with the correct arguments
    mock_import_module.assert_called_once_with("module_foo", "lazy_module")

    # Verify the attribute's value
    assert foo_value == "foo_value"


def test_lazy_import_not_in_dict(lazy_module):
    with pytest.raises(AttributeError):
        _ = getattr(lazy_module, "non_existent_attribute")


def test_dir_includes_lazy_imports(lazy_module):
    dir_list = lazy_module.__dir__()
    assert "foo" in dir_list
    assert "bar" in dir_list


def test_all_property(lazy_module):
    assert lazy_module.__all__ == ["foo", "bar"]
