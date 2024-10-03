import inspect
from typing import AsyncIterable, Iterable
from unittest.mock import MagicMock, call

import datasets
import pytest

from hyped.core.flow import DataFlow
from hyped.nodes.decorators import _create_in_out_types, as_aggregator, as_augmenter, as_processor


def test_create_in_out_types():
    def func(a: int, b: str) -> float:
        ...

    signature = inspect.signature(func)
    input_refs_t, output_refs_t = _create_in_out_types(
        "prefix", signature.parameters, signature.return_annotation
    )

    assert "a" in input_refs_t.__annotations__
    assert "b" in input_refs_t.__annotations__
    assert "output" in output_refs_t.model_fields


def test_create_in_out_types_errors():
    def func(a: int, b: str, **kwargs) -> float:
        ...

    signature = inspect.signature(func)
    with pytest.raises(ValueError):
        _create_in_out_types("prefix", signature.parameters, signature.return_annotation)

    def func(a, b: str) -> float:
        ...

    signature = inspect.signature(func)
    with pytest.raises(ValueError):
        _create_in_out_types("prefix", signature.parameters, signature.return_annotation)

    def func(a: int, b: str):
        ...

    signature = inspect.signature(func)
    with pytest.raises(ValueError):
        _create_in_out_types("prefix", signature.parameters, signature.return_annotation)


class TestAsProcessor:
    def test_as_processor(self):
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

    def test_async_as_processor(self):
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


class TestAsAugmenter:
    def test_as_augmenter(self):
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

    def test_async_as_augmenter(self):
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

    def test_errors(self):
        with pytest.raises(ValueError):
            # not a generator function
            @as_augmenter
            def func(a: int, b: str, **kwargs) -> float:  # noqa: F811
                ...

        with pytest.raises(ValueError):
            # generator function but incorrect return type hint
            @as_augmenter
            def func(a: int, b: str, **kwargs) -> float:  # noqa: F811
                yield 0.0

        with pytest.raises(ValueError):
            # generator function but incomplete return type hint
            @as_augmenter
            def func(a: int, b: str, **kwargs) -> Iterable:  # noqa: F811
                yield 0.0


class TestAsAggregator:
    def test_as_aggregator(self):
        # create a mock object to track the function calls
        mock = MagicMock()

        @as_aggregator
        def mock_func(value: None | int, ctx: None | int, x: int) -> tuple[int, int]:
            mock(x)
            value = value or 0
            ctx = ctx or 0
            return x + value, ctx + 1

        # create a dummy dataset
        features = datasets.Features({"x": datasets.Value("int32")})
        ds = datasets.Dataset.from_dict({"x": range(10)}, features=features)
        # create flow and apply to dummy dataset
        flow = DataFlow(ds.features)
        out = mock_func(x=flow.src_features.x)
        _, agg = flow.apply(ds, collect=flow.src_features, aggregate={"out": out})
        # check mock object has been called for each sample and output matches expectation
        mock.assert_has_calls(calls=[call(i) for i in range(10)], any_order=True)
        assert agg["out"] == sum(range(10))

    def test_async_as_aggregator(self):
        # create a mock object to track the function calls
        mock = MagicMock()

        @as_aggregator
        async def mock_func(value: None | int, ctx: None | int, x: int) -> tuple[int, int]:
            mock(x)
            value = value or 0
            ctx = ctx or 0
            return x + value, ctx + 1

        # create a dummy dataset
        features = datasets.Features({"x": datasets.Value("int32")})
        ds = datasets.Dataset.from_dict({"x": range(10)}, features=features)
        # create flow and apply to dummy dataset
        flow = DataFlow(ds.features)
        out = mock_func(x=flow.src_features.x)
        _, agg = flow.apply(ds, collect=flow.src_features, aggregate={"out": out})
        # check mock object has been called for each sample and output matches expectation
        mock.assert_has_calls(calls=[call(i) for i in range(10)], any_order=True)
        assert agg["out"] == sum(range(10))

    def test_errors(self):
        with pytest.raises(ValueError):
            # missing value argument
            @as_aggregator
            def func(ctx: None | int, x: int) -> tuple[int, int]:  # noqa: F811
                ...

        with pytest.raises(ValueError):
            # missing context argument
            @as_aggregator
            def func(value: None | int, x: int) -> tuple[int, int]:  # noqa: F811
                ...

        with pytest.raises(ValueError):
            # value argument is not optional
            @as_aggregator
            def func(value: int, ctx: None | int, x: int) -> tuple[int, int]:  # noqa: F811
                ...

        with pytest.raises(ValueError):
            # context argument is not optional
            @as_aggregator
            def func(value: None | int, ctx: int, x: int) -> tuple[int, int]:  # noqa: F811
                ...

        with pytest.raises(ValueError):
            # mismatch between value and return type
            @as_aggregator
            def func(value: None | int, ctx: int, x: int) -> tuple[float, int]:  # noqa: F811
                ...

        with pytest.raises(ValueError):
            # mismatch between context and return type
            @as_aggregator
            def func(value: None | int, ctx: int, x: int) -> tuple[int, float]:  # noqa: F811
                ...

        with pytest.raises(ValueError):
            # mismatch between context and return type
            @as_aggregator
            def func(value: None | int, ctx: None, x: int) -> tuple[int, int]:  # noqa: F811
                ...

        with pytest.raises(ValueError):
            # wrong return type
            @as_aggregator
            def func(value: None | int, ctx: None, x: int) -> int:  # noqa: F811
                ...
