"""This module defines a collection of data processors for common feature operations."""

from typing import Annotated, TypeVar

import numpy as np
import pyarrow.compute as pc

from ..features.features import Feature as _Feature
from ..features.validators import MatchFeatures
from ..nodes.augmentor import BaseDataAugmentor, BaseDataAugmentorConfig
from ..nodes.base import RunContext, process_mode
from ..nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from ..typing import Bool, Feature, TraceIndexList


class EqConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`Eq` processor."""


T = Annotated[TypeVar("T", bound=Feature), MatchFeatures()]


class Eq(BaseDataProcessor[EqConfig]):
    """Data processor for checking equality between two feature values."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, a: T, b: T) -> Bool:
        """Computes whether two feature values are equal.

        Args:
            ctx (RunContext): The execution context.
            a (T): The first feature value.
            b (T): The second feature value.

        Returns:
            Bool: :code:`True` if the feature values are equal, otherwise :code:`False`.
        """
        return pc.equal(a, b)


class NotEqConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`NotEq` processor."""


class NotEq(BaseDataProcessor[NotEqConfig]):
    """Data processor for checking inverse equality between two feature values."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, a: T, b: T) -> Bool:
        """Computes whether two feature values are not equal.

        Args:
            ctx (RunContext): The execution context.
            a (T): The first feature value.
            b (T): The second feature value.

        Returns:
            Bool: :code:`False` if the feature values are equal, otherwise :code:`True`.
        """
        return pc.not_equal(a, b)


class GlobalFilterConfig(BaseDataAugmentorConfig):
    """Configuration for the :class:`GlobalFilter` augmentor."""


T = TypeVar("T", bound=Feature)


class GlobalFilter(BaseDataAugmentor[GlobalFilterConfig]):
    """Data augmentor for filtering values based on a condition.

    This processor filters input values using a specified boolean condition,
    returning the filtered values along with their trace indices.
    """

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, value: T, condition: Bool) -> tuple[T, TraceIndexList]:
        """Filters input values based on a boolean condition.

        Args:
            ctx (RunContext): The execution context.
            value (T): The input values to filter.
            condition (Bool): A boolean array indicating which elements to retain.

        Returns:
            tuple[T, TraceIndexList]:
                - The filtered values that satisfy the condition.
                - The indices of the filtered values in the original array.
        """
        # compute the trace indices and the filtered values
        (trace_indices,) = np.nonzero(condition.to_numpy())
        filtered_value = pc.filter(value, condition)
        return filtered_value, trace_indices


# register methods
_Feature.register_method("__eq__")(Eq().call)
_Feature.register_method("__ne__")(NotEq().call)


@_Feature.register_method("filter")
def filter_(value: T, condition: Bool) -> T:
    """Filters input values based on a boolean condition using the :class:`GlobalFilter` augmentor.

    Args:
        value (T): The input values to filter.
        condition (Bool): A boolean array indicating which elements to retain.

    Returns:
        T: The filtered values that satisfy the given condition.
    """
    return GlobalFilter().call(value=value, condition=condition)
