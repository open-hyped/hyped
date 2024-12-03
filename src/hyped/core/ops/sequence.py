"""This module defines a collection of data processors that implement common sequence operations.

Each processor is designed to handle a specific sequence transformation or query, such as
calculating lengths, slicing or aggregation operations. These processors are intended for
use in data processing pipelines, where they can be applied in a batched and efficient
manner using Apache Arrow as the backend.

These processors are registered as methods on the :class:`SequenceFeature` class, allowing them
to be applied directly to sequence features.
"""

from typing import Any, TypeVar

import pyarrow.compute as pc

from ..features.features import Int32Feature, SequenceFeature
from ..nodes.base import RunContext, process_mode
from ..nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from ..typing import Float, Int, Sequence, UInt


class LengthConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`Length` processor."""


class Length(BaseDataProcessor[LengthConfig]):
    """Data processor for computing the length of sequences."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, x: Sequence) -> Int32Feature:
        """Computes the length of an input sequence.

        Args:
            ctx (RunContext): The execution context.
            x (Sequence): The sequence to compute the length for.

        Returns:
            IntFeature: The lengths of the input sequences.
        """
        return pc.list_value_length(x)


NumericType = TypeVar("T", bound=Int | Float | UInt)


class SequenceMinConfig(BaseDataProcessorConfig):
    """Configuration class for the SequenceMin processor."""

    default: None | int | float = None
    """The default value to return if the sequence is empty."""


class SequenceMin(BaseDataProcessor[SequenceMinConfig]):
    """Processor to compute the minimum value of a numeric sequence."""

    @process_mode(batched=False, backend="python")
    def process(self, ctx: RunContext, x: Sequence[NumericType]) -> NumericType:
        """Compute the minimum value of a sequence.

        Args:
            ctx (RunContext): Context object containing runtime information.
            x (Sequence[NumericType]): Input sequence of numeric values.

        Returns:
            NumericType: The minimum value in the sequence, or the default value
            specified in the configuration if the sequence is empty.
        """
        return min(x, default=self.config.default)


class SequenceMaxConfig(BaseDataProcessorConfig):
    """Configuration class for the SequenceMax processor."""

    default: None | int | float = None
    """The default value to return if the sequence is empty."""


class SequenceMax(BaseDataProcessor[SequenceMaxConfig]):
    """Processor to compute the maximum value of a numeric sequence."""

    @process_mode(batched=False, backend="python")
    def process(self, ctx: RunContext, x: Sequence[NumericType]) -> NumericType:
        """Compute the maximum value of a sequence.

        Args:
            ctx (RunContext): Context object containing runtime information.
            x (Sequence[NumericType]): Input sequence of numeric values.

        Returns:
            NumericType: The maximum value in the sequence, or the default value
            specified in the configuration if the sequence is empty.
        """
        return max(x, default=self.config.default)


class SequenceSumConfig(BaseDataProcessorConfig):
    """Configuration class for the SequenceSum processor."""


class SequenceSum(BaseDataProcessor[SequenceSumConfig]):
    """Processor to compute the sum of numeric values in a sequence."""

    @process_mode(batched=False, backend="python")
    def process(self, ctx: RunContext, x: Sequence[NumericType]) -> NumericType:
        """Compute the sum of a sequence.

        Args:
            ctx (RunContext): Context object containing runtime information.
            x (Sequence[NumericType]): Input sequence of numeric values.

        Returns:
            NumericType: The sum of the values in the sequence, or the default value
            specified in the configuration if the sequence is empty.
        """
        return sum(x)


@SequenceFeature.register_method("min")
def sequence_min(seq: Sequence[NumericType], default: Any = None) -> NumericType:
    """Compute the minimum value of a sequence using the :class:`SequenceMin` processor.

    Args:
        seq (Sequence[NumericType]): Input sequence feature of numeric values.
        default (Any): Default value to return if the sequence is empty. Defaults to :code:`None`.

    Returns:
        NumericType: The feature representing the minimum value in the sequence, or the provided
        default value if the sequence is empty.
    """
    return SequenceMin(default=default).call(seq)


@SequenceFeature.register_method("max")
def sequence_max(seq: Sequence[NumericType], default: Any = None) -> NumericType:
    """Compute the maximum value of a sequence using the :class:`SequenceMax` processor.

    Args:
        seq (Sequence[NumericType]): Input sequence feature of numeric values.
        default (Any): Default value to return if the sequence is empty. Defaults to :code:`None`.

    Returns:
        NumericType: The feature representing the maximum value in the sequence, or the provided
        default value if the sequence is empty.
    """
    return SequenceMax(default=default).call(seq)


SequenceFeature.register_method("sum")(SequenceSum().call)
SequenceFeature.register_method("length")(Length().call)
