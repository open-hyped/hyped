from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hyped.core.features.dtypes import (
    Float32Type,
    Int32Type,
    Int64Type,
    StringType,
    Type,
    UInt8Type,
)
from hyped.core.features.features import Feature
from hyped.core.ops.numeric import (
    Abs,
    Add,
    FloorDiv,
    Multiply,
    Negate,
    Subtract,
    TrueDiv,
    add_constant,
    handle_constant_for_binary_operation,
)
from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.core.typing import Float, Float64, Int, UInt


@pytest.mark.parametrize(
    "val, candidate_dtype, expected_dtype",
    [
        (5, Int32Type, Int32Type),
        (5.0, Float32Type, Float32Type),
        (5, Float32Type, Int32Type),
        (5, UInt8Type, UInt8Type),
        (5, Int32Type, Int32Type),
        (5, Int64Type, Int64Type),
        (-5, Int32Type, Int32Type),
        (-5, UInt8Type, Int32Type),
        ("string", StringType, StringType),
    ],
)
def test_add_constant(val: Any, candidate_dtype: Type, expected_dtype: Type) -> None:
    graph = MagicMock()
    # Call the add_constant function
    add_constant(val, candidate_dtype, graph)
    graph.add_const_node.assert_called_once_with(val, expected_dtype)


@patch("hyped.core.ops.numeric.add_constant")
def test_handle_constant_for_binary_operation(mock_add_constant: MagicMock) -> None:
    a = Feature(MagicMock(), MagicMock())
    b = Feature(MagicMock(), MagicMock())
    c = MagicMock()
    f = MagicMock()
    # apply decorator to mock function
    wrapped = handle_constant_for_binary_operation(f)
    # call wrapped function and check call to mock function
    wrapped(a, b)
    f.assert_called_with(a, b)
    # call wrapped function with constant right input
    wrapped(a, c)
    mock_add_constant.assert_called_with(c, a.dtype, a.ref._graph)
    f.assert_called_with(a, mock_add_constant.return_value)
    # call wrapped function with constant left input
    wrapped(c, b)
    mock_add_constant.assert_called_with(c, b.dtype, b.ref._graph)
    f.assert_called_with(mock_add_constant.return_value, b)


class TestInvertInt(BaseDataProcessorTest):
    processor = Negate()
    input_features = {"x": Int}
    input_data = [{"x": 5}, {"x": -5}]
    expected_output_feature = Int
    expected_output_data = [-5, 5]


class TestInvertUInt(BaseDataProcessorTest):
    processor = Negate()
    input_features = {"x": UInt}
    input_data = [{"x": 5}, {"x": 10}]
    expected_output_feature = Int
    expected_output_data = [-5, -10]


class TestInvertFloat(BaseDataProcessorTest):
    processor = Negate()
    input_features = {"x": Float}
    input_data = [{"x": 5.0}, {"x": -5.0}]
    expected_output_feature = Float
    expected_output_data = [-5.0, 5.0]


class TestAbs(BaseDataProcessorTest):
    processor = Abs()
    input_features = {"x": Int}
    input_data = [{"x": 5}, {"x": -5}]
    expected_output_feature = Int
    expected_output_data = [5, 5]


class TestAdd(BaseDataProcessorTest):
    processor = Add()
    input_features = {"x": Int, "y": Int}
    input_data = [{"x": 5, "y": 5}, {"x": -5, "y": 5}]
    expected_output_feature = Int
    expected_output_data = [10, 0]


class TestSubtract(BaseDataProcessorTest):
    processor = Subtract()
    input_features = {"x": Int, "y": Int}
    input_data = [{"x": 5, "y": 3}, {"x": -5, "y": 5}]
    expected_output_feature = Int
    expected_output_data = [2, -10]


class TestMultiply(BaseDataProcessorTest):
    processor = Multiply()
    input_features = {"x": Int, "y": Int}
    input_data = [{"x": 5, "y": 3}, {"x": -5, "y": 5}]
    expected_output_feature = Int
    expected_output_data = [15, -25]


class TestTrueDiv(BaseDataProcessorTest):
    processor = TrueDiv()
    input_features = {"x": Int, "y": Int}
    input_data = [{"x": 10, "y": 2}, {"x": -10, "y": 5}]
    expected_output_feature = Float64
    expected_output_data = [5, -2]


class TestFloorDiv(BaseDataProcessorTest):
    processor = FloorDiv()
    input_features = {"x": Int, "y": Int}
    input_data = [{"x": 10, "y": 3}, {"x": -10, "y": 4}]
    expected_output_feature = Int
    expected_output_data = [3, -2]
