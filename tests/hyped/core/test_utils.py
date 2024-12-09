from unittest.mock import MagicMock

from hyped.core.utils import map_recursive


def test_map_recursive():
    # Define a nested structure with various levels of dictionaries and lists
    nested_obj = {
        "key1": [1, 2, {"key2": 3}],
        "key3": {"key4": [4, 5], "key5": 6},
    }

    # Mock function to apply
    mock_fn = MagicMock(side_effect=lambda path, x: x)

    # Apply map_recursive
    result = map_recursive(mock_fn, nested_obj)

    # Verify the structure remains unchanged (since fn returns x unchanged)
    assert result == nested_obj

    # Verify the mock function was called for every element in the nested object
    expected_calls = [
        ((tuple(), nested_obj),),
        # Calls for top-level keys
        ((("key1",), [1, 2, {"key2": 3}]),),
        ((("key3",), {"key4": [4, 5], "key5": 6}),),
        # Calls for elements inside key1
        ((("key1", 0), 1),),
        ((("key1", 1), 2),),
        ((("key1", 2), {"key2": 3}),),
        ((("key1", 2, "key2"), 3),),
        # Calls for elements inside key3
        ((("key3", "key4"), [4, 5]),),
        ((("key3", "key4", 0), 4),),
        ((("key3", "key4", 1), 5),),
        ((("key3", "key5"), 6),),
    ]

    mock_fn.assert_has_calls(expected_calls, any_order=True)

    # Verify the number of calls matches the number of elements in the structure
    assert mock_fn.call_count == len(expected_calls)
