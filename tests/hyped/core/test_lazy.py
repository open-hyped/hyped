from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

import pytest

from hyped.common.utils import deep_equal
from hyped.core.lazy import LazyFlowOutput


@pytest.fixture
def mock_executor():
    executor = MagicMock()
    executor.execute = AsyncMock(return_value={"y": [1]})
    executor.collect.feature_.keys = MagicMock(return_value=["y"])
    return executor


def test_lazy_flow_initialization(mock_executor):
    input_proxy = MappingProxyType({"x": 0})
    obj = LazyFlowOutput(input_proxy, mock_executor)

    assert obj._proxy == input_proxy
    assert obj._executor == mock_executor
    assert obj._proxy_snapshot is None
    assert obj._out_snapshot is None


def test_lazy_flow_keys(mock_executor):
    input_proxy = MappingProxyType({"x": 0})
    obj = LazyFlowOutput(input_proxy, mock_executor)

    assert list(obj.keys()) == ["y"]
    mock_executor.collect.feature_.keys.assert_called_once()


def test_lazy_flow_getitem(mock_executor):
    input_dict = {"x": 0}
    input_proxy = MappingProxyType(input_dict)

    obj = LazyFlowOutput(input_proxy, mock_executor)
    prev_snapshot = obj._proxy_snapshot

    # request a value
    assert obj["y"] == 1

    mock_executor.execute.assert_called_once_with({"x": [input_dict["x"]]}, index=[0], rank=0)
    assert obj._out_snapshot == {"y": 1}
    assert not deep_equal(obj._proxy_snapshot, prev_snapshot)

    # reset the executor mock
    mock_executor.reset_mock()

    # request another value without changing the inputs
    assert obj["y"] == 1
    assert not mock_executor.execute.called

    # change the input and make sure the flow is executed again
    input_dict["x"] = 1
    prev_snapshot = obj._proxy_snapshot

    # request a value
    assert obj["y"] == 1

    mock_executor.execute.assert_called_once_with({"x": [input_dict["x"]]}, index=[0], rank=0)
    assert obj._proxy_snapshot != prev_snapshot


def test_lazy_flow_getitem_keyerror(mock_executor):
    input_proxy = MappingProxyType({"x": 0})
    obj = LazyFlowOutput(input_proxy, mock_executor)

    try:
        obj["z"]
    except KeyError as e:
        assert str(e) == "'z'"


def test_lazy_flow_iteration(mock_executor):
    input_proxy = MappingProxyType({"x": 0})
    obj = LazyFlowOutput(input_proxy, mock_executor)
    keys = list(iter(obj))

    assert keys == ["y"]


def test_lazy_flow_length(mock_executor):
    input_proxy = MappingProxyType({"x": 0})
    obj = LazyFlowOutput(input_proxy, mock_executor)
    assert len(obj) == 1


def test_lazy_flow_str(mock_executor):
    input_proxy = MappingProxyType({"x": 0})
    obj = LazyFlowOutput(input_proxy, mock_executor)
    _ = obj["y"]

    assert str(obj) == str({"y": 1})


def test_lazy_flow_repr(mock_executor):
    input_proxy = MappingProxyType({"x": 0})
    obj = LazyFlowOutput(input_proxy, mock_executor)

    assert repr(obj) == f"LazyFlowOutput(input_proxy={input_proxy}, executor={mock_executor})"
