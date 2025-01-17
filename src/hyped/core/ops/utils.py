"""This module defines a collection of data processors that implement utility operations."""

from typing import TypeVar

import numpy as np
import pyarrow.compute as pc

from ..features.features import Feature as _Feature
from ..nodes.augmentor import BaseDataAugmentor, BaseDataAugmentorConfig
from ..nodes.base import RunContext, process_mode
from ..typing import Bool, Feature, TraceIndexList


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


@_Feature.register_method("filter")
def filter_(value: T, condition: Bool) -> T:
    """Filters input values based on a boolean condition using the `GlobalFilter` augmentor.

    This function acts as a convenience wrapper for applying the `GlobalFilter` augmentor.
    It filters the provided input values using the specified boolean condition and returns
    the filtered values.

    Args:
        value (T): The input values to filter.
        condition (Bool): A boolean array indicating which elements to retain.

    Returns:
        T: The filtered values that satisfy the given condition.
    """
    return GlobalFilter().call(value=value, condition=condition)
