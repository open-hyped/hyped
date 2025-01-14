"""This module defines the :class:`TraceNode`.

The :class:`TraceNode` class is a specialized node designed to apply trace indices through a
specified path in a partition graph. It transforms input values by tracing the sequence of
partition transitions and applying the corresponding trace indices at each step.
"""

from typing import Any

import numpy as np
import pyarrow as pa

from ..typing import PartitionId
from .base import BaseNode, BaseNodeConfig, RunContext


class TraceNodeConfig(BaseNodeConfig):
    """Configuration for a :class:`TraceNode`."""

    path: tuple[PartitionId, ...]
    """The path through the partition graph

    A tuple representing the ordered sequence of partition identifiers that
    define the path in the partition graph.
    """


class TraceNode(BaseNode[TraceNodeConfig]):
    """A node that applies trace indices through a specified partition path.

    This node traces a path in the partition graph as specified in the configuration
    and applies the trace indices for each transition between partitions to transform
    the input values.
    """

    def trace_values_through_partition_path(
        self,
        ctx: RunContext,
        values: pa.Array,
        traces: dict[tuple[PartitionId, PartitionId], np.ndarray],
    ) -> pa.Array:
        """Apply trace indices through the partition path to transform the provided values.

        This method traces the specified path in the partition graph and applies the trace
        indices of each partition transition to generate a final trace index. The resulting
        index is used to transform the provided values.

        Args:
            ctx (RunContext): The runtime context.
            values (pa.Array): The values to be transformed.
            traces (dict[tuple[PartitionId, PartitionId], np.ndarray]): A dictionary mapping
                partition transitions to their corresponding trace indices.

        Returns:
            pa.Array: A transformed PyArrow array where values have been modified
                according to the trace indices along the specified path.
        """
        trace_index = np.arange(len(ctx.index))
        # follow the path from partition u to partition v and apply the trace
        # of each partition transition to build the final trace index
        for edge in zip(self.config.path[:-1], self.config.path[1:], strict=True):
            trace_index = trace_index[traces[edge]]

        # apply the final trace index to the given values
        return values.take(trace_index)

    @property
    def signature(self) -> Any:  # pragma: not covered
        """Raises an error, as signature is not supported for this node."""
        raise EnvironmentError("The `signature` property is not available for trace nodes.")

    def call(self, *args: Any, **kwargs: Any) -> Any:  # pragma: not covered
        """Raises an error, as direct calls are not supported for this node."""
        raise EnvironmentError("The `call` method is not available for trace nodes.")
