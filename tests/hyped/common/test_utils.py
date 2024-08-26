import pytest

from hyped.common.utils import (
    dict_of_lists_to_list_of_dicts,
    list_of_dicts_to_dict_of_lists,
)


def test_list_of_dicts_to_dict_of_lists():
    # Test case 1: Basic functionality
    dicts = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    keys = ["a", "b"]
    expected = {"a": [1, 3], "b": [2, 4]}
    assert list_of_dicts_to_dict_of_lists(dicts, keys) == expected

    # Test case 2: Default value when a key is missing
    dicts = [{"a": 1}, {"b": 4}]
    keys = ["a", "b"]
    expected = {"a": [1, None], "b": [None, 4]}
    assert list_of_dicts_to_dict_of_lists(dicts, keys) == expected

    # Test case 3: Custom default value
    dicts = [{"a": 1}, {"b": 4}]
    keys = ["a", "b"]
    expected = {"a": [1, 0], "b": [0, 4]}
    assert list_of_dicts_to_dict_of_lists(dicts, keys, default=0) == expected


def test_dict_of_lists_to_list_of_dicts():
    # Test case 1: Basic functionality
    dict_of_lists = {"a": [1, 3], "b": [2, 4]}
    expected = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert dict_of_lists_to_list_of_dicts(dict_of_lists) == expected

    # Test case 2: Empty dictionary
    dict_of_lists = {}
    expected = []
    assert dict_of_lists_to_list_of_dicts(dict_of_lists) == expected

    # Test case 3: Lists of different sizes should raise an AssertionError
    dict_of_lists = {"a": [1, 3], "b": [2]}
    with pytest.raises(AssertionError):
        dict_of_lists_to_list_of_dicts(dict_of_lists)
