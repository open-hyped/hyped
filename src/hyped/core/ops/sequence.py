"""This module defines a collection of data processors that implement common sequence operations.

Each processor is designed to handle a specific sequence transformation or query, such as
calculating lengths, slicing or aggregation operations. These processors are intended for
use in data processing pipelines, where they can be applied in a batched and efficient
manner using Apache Arrow as the backend.

These processors are registered as methods on the :class:`SequenceFeature` class, allowing them
to be applied directly to sequence features.
"""


import pyarrow.compute as pc

from ..features.features import Int32Feature, SequenceFeature
from ..nodes.base import RunContext, process_mode
from ..nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from ..typing import Sequence


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


SequenceFeature.register_method("length")(Length().call)
