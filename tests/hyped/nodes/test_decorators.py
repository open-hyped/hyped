from typing import AsyncIterable, Iterable
from unittest.mock import MagicMock, call

import datasets

from hyped.core.flow import DataFlow
from hyped.nodes.decorators import as_augmenter, as_processor


def test_as_processor():
    # create a mock object to track the function calls
    mock = MagicMock()

    @as_processor
    def mock_func(x: int) -> int:
        mock(x)
        return x * 2

    # create a dummy dataset
    features = datasets.Features({"x": datasets.Value("int32")})
    ds = datasets.Dataset.from_dict({"x": range(10)}, features=features)
    # create flow and apply to dummy dataset
    flow = DataFlow(ds.features)
    out = mock_func(x=flow.src_features.x)
    out_ds, _ = flow.apply(ds, collect={"out": out})
    # check mock object has been called for each sample and output matches expectation
    mock.assert_has_calls(calls=[call(i) for i in range(10)], any_order=True)
    assert all(2 * i == j for i, j in zip(ds["x"], out_ds["out"]))


def test_async_as_processor():
    # create a mock object to track the function calls
    mock = MagicMock()

    @as_processor
    async def mock_func(x: int) -> int:
        mock(x)
        return x * 2

    # create a dummy dataset
    features = datasets.Features({"x": datasets.Value("int32")})
    ds = datasets.Dataset.from_dict({"x": range(10)}, features=features)
    # create flow and apply to dummy dataset
    flow = DataFlow(ds.features)
    out = mock_func(x=flow.src_features.x)
    out_ds, _ = flow.apply(ds, collect={"out": out})
    # check mock object has been called for each sample and output matches expectation
    mock.assert_has_calls(calls=[call(i) for i in range(10)], any_order=True)
    assert all(2 * i == j for i, j in zip(ds["x"], out_ds["out"]))


def test_as_augmenter():
    # create a mock object to track the function calls
    mock = MagicMock()

    @as_augmenter
    def mock_func(x: int) -> Iterable[int]:
        mock(x)
        yield x * 2
        yield x * 3

    # create a dummy dataset
    features = datasets.Features({"x": datasets.Value("int32")})
    ds = datasets.Dataset.from_dict({"x": range(10)}, features=features)
    # create flow and apply to dummy dataset
    flow = DataFlow(ds.features)
    out = mock_func(x=flow.src_features.x)
    out_ds, _ = flow.apply(ds, collect={"out": out})
    # check mock object has been called for each sample and output matches expectation
    mock.assert_has_calls(calls=[call(i) for i in range(10)], any_order=True)
    assert len(out_ds) == 2 * len(ds)


def test_async_as_augmenter():
    # create a mock object to track the function calls
    mock = MagicMock()

    @as_augmenter
    async def mock_func(x: int) -> AsyncIterable[int]:
        mock(x)
        yield x * 2
        yield x * 3

    # create a dummy dataset
    features = datasets.Features({"x": datasets.Value("int32")})
    ds = datasets.Dataset.from_dict({"x": range(10)}, features=features)
    # create flow and apply to dummy dataset
    flow = DataFlow(ds.features)
    out = mock_func(x=flow.src_features.x)
    out_ds, _ = flow.apply(ds, collect={"out": out})
    # check mock object has been called for each sample and output matches expectation
    mock.assert_has_calls(calls=[call(i) for i in range(10)], any_order=True)
    assert len(out_ds) == 2 * len(ds)
