import sys
from unittest.mock import patch

import pytest

from hyped.common.utils import is_package_installed, is_python_version_less_than, tmp_setattr


class DummyObject:
    def __init__(self) -> None:
        self.attr = "original"


@pytest.fixture
def dummy_instance():
    return DummyObject()


def test_tmp_setattr_restore_existing_attribute(dummy_instance: object) -> None:
    assert dummy_instance.attr == "original"  # Initial value
    with tmp_setattr(dummy_instance, "attr", "temporary"):
        assert dummy_instance.attr == "temporary"  # Temporary value
    assert dummy_instance.attr == "original"  # Restored to original


def test_tmp_setattr_add_new_attribute(dummy_instance: object) -> None:
    assert not hasattr(dummy_instance, "new_attr")  # New attribute doesn't exist
    with tmp_setattr(dummy_instance, "new_attr", "temporary"):
        assert dummy_instance.new_attr == "temporary"  # Temporary value
    assert not hasattr(dummy_instance, "new_attr")  # Removed after context


def test_tmp_setattr_restore_deleted_attribute(dummy_instance: object) -> None:
    del dummy_instance.attr  # Delete the attribute
    assert not hasattr(dummy_instance, "attr")  # Ensure it's deleted
    with tmp_setattr(dummy_instance, "attr", "temporary"):
        assert dummy_instance.attr == "temporary"  # Temporary value
    assert not hasattr(dummy_instance, "attr")  # Attribute remains deleted after context


def test_tmp_setattr_handle_exceptions(dummy_instance: object) -> None:
    assert dummy_instance.attr == "original"  # Initial value
    with pytest.raises(ValueError):
        with tmp_setattr(dummy_instance, "attr", "temporary"):
            assert dummy_instance.attr == "temporary"  # Temporary value
            raise ValueError("Intentional error")  # Raise an exception
    assert dummy_instance.attr == "original"  # Restored to original even after exception


def test_is_package_installed_existing_package() -> None:
    # Mock `importlib.util.find_spec` to return a mock object for an existing package
    with patch("importlib.util.find_spec", return_value=object()):
        # Test for a package that exists (mocked)
        assert is_package_installed("existing_package") is True


def test_is_package_installed_non_existing_package() -> None:
    # Mock `importlib.util.find_spec` to return None for a non-existing package
    with patch("importlib.util.find_spec", return_value=None):
        # Test for a package that does not exist
        assert is_package_installed("non_existing_package") is False


def test_is_package_installed_real_package() -> None:
    # Test with a package that is actually installed (e.g., 'pytest')
    assert is_package_installed("pytest") is True


def test_is_package_installed_real_non_existing_package() -> None:
    # Test with a package that is very unlikely to exist
    assert is_package_installed("some_non_existing_random_package") is False


@pytest.mark.parametrize(
    "current_version, target_version, expected",
    [
        ((3, 8, 0), (3, 9, 0), True),
        ((3, 9, 0), (3, 9, 0), False),
        ((3, 10, 0), (3, 9, 0), False),
        ((3, 9, 1), (3, 9, 2), True),
        ((3, 9, 2), (3, 9, 1), False),
        ((3, 9, 2), (3, 9, 2), False),
        ((3, 9, 0), (4, 0, 0), True),
    ],
)
def test_is_python_version_less_than(current_version, target_version, expected):
    major, minor, micro = target_version
    with patch.object(sys, "version_info", current_version):
        assert is_python_version_less_than(major, minor, micro) == expected
