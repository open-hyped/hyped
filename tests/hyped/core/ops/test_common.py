from unittest.mock import MagicMock, patch

from hyped.core.ops.common import Eq, GlobalFilter, NotEq, filter_
from hyped.core.testing.augmentor import BaseDataAugmentorTest
from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.typing import Bool, Int


class TestEq(BaseDataProcessorTest):
    processor = Eq()
    input_features = {"a": Bool, "b": Bool}
    input_data = [
        # True
        {"a": True, "b": True},
        {"a": False, "b": False},
        # False
        {"a": False, "b": True},
    ]
    expected_output_feature = Bool
    expected_output_data = [True, True, False]


class TestNotEqu(BaseDataProcessorTest):
    processor = NotEq()
    input_features = {"a": Bool, "b": Bool}
    input_data = [
        # True
        {"a": True, "b": True},
        {"a": False, "b": False},
        # False
        {"a": False, "b": True},
    ]
    expected_output_feature = Bool
    expected_output_data = [False, False, True]


class TestGlobalFilter(BaseDataAugmentorTest):
    augmentor = GlobalFilter()
    input_features = {"value": Int, "condition": Bool}
    input_data = [
        {"value": 0, "condition": True},
        {"value": 1, "condition": True},
        {"value": 2, "condition": False},
        {"value": 3, "condition": False},
        {"value": 4, "condition": True},
    ]
    expected_output_feature = Int
    expected_output_data = [0, 1, 4]


class TestGlobalFilterAllFalse(BaseDataAugmentorTest):
    augmentor = GlobalFilter()
    input_features = {"value": Int, "condition": Bool}
    input_data = [
        {"value": 0, "condition": False},
        {"value": 1, "condition": False},
        {"value": 2, "condition": False},
        {"value": 3, "condition": False},
        {"value": 4, "condition": False},
    ]
    expected_output_feature = Int
    expected_output_data = []


class TestGlobalFilterAllTrue(BaseDataAugmentorTest):
    augmentor = GlobalFilter()
    input_features = {"value": Int, "condition": Bool}
    input_data = [
        {"value": 0, "condition": True},
        {"value": 1, "condition": True},
        {"value": 2, "condition": True},
        {"value": 3, "condition": True},
        {"value": 4, "condition": True},
    ]
    expected_output_feature = Int
    expected_output_data = [0, 1, 2, 3, 4]


def test_filter_uses_global_filter():
    # Mock inputs
    mock_value = MagicMock(name="value")
    mock_condition = MagicMock(name="condition")
    mock_filtered_value = MagicMock(name="filtered_value")

    # Patch GlobalFilter to mock its behavior
    with patch("hyped.core.ops.common.GlobalFilter") as MockGlobalFilter:
        # Configure the mock GlobalFilter instance
        mock_instance = MockGlobalFilter.return_value
        mock_instance.call.return_value = mock_filtered_value

        # Call the filter_ function
        result = filter_(mock_value, mock_condition)

        # Assert that GlobalFilter was instantiated
        MockGlobalFilter.assert_called_once_with()

        # Assert that the call method of GlobalFilter was called with the correct arguments
        mock_instance.call.assert_called_once_with(value=mock_value, condition=mock_condition)

        # Assert that the result matches the mocked filtered value
        assert result == mock_filtered_value
