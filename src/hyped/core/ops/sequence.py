"""This module defines a collection of data processors that implement common sequence operations.

Each processor is designed to handle a specific sequence transformation or query, such as
calculating lengths, slicing or aggregation operations. These processors are intended for
use in data processing pipelines, where they can be applied in a batched and efficient
manner using Apache Arrow as the backend.

These processors are registered as methods on the :class:`SequenceFeature` class, allowing them
to be applied directly to sequence features.
"""

from typing import Any, Generic, TypeVar

import pyarrow as pa
import pyarrow.compute as pc

from hyped.common._pyarrow import flatten_list_array, unflatten_list_array
from hyped.core.typing import PartitionId

from ..features.features import Int32Feature, SequenceFeature
from ..features.reference import Reference
from ..graph import DataFlowGraph
from ..nodes.augmenter import BaseDataAugmenter, BaseDataAugmenterConfig
from ..nodes.base import RunContext, process_mode
from ..nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from ..typing import Float, Int, Int32, Mapping, Sequence, TraceIndexList, UInt


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


T = TypeVar("T")


class FlatValuesWithIndex(Mapping, Generic[T]):
    """Represents a flat sequence values with associated index mapping."""

    value: T
    """The values of the flattened sequence."""

    index: Int32Feature
    """The index mapping of the original sequence elements."""


class SequenceUnpackConfig(BaseDataAugmenterConfig):
    """Configuration class for the :class:`SequenceUnpack` augmenter."""


class SequenceUnpack(BaseDataAugmenter[SequenceUnpackConfig]):
    """Augmenter to unpack a sequence."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, seq: Sequence[T]) -> tuple[T, TraceIndexList]:
        """Unpack the sequence.

        Args:
            ctx (RunContext): Context object containing runtime information.
            seq (Sequence[T]): Input sequence to be flattened.

        Returns:
            tuple[FlatSequenceWithIndex[T], TraceIndexList]: A tuple containing the flattened
            sequence with indices and the computed trace indices.
        """
        # flatten the sequence and compute the trace indices
        flattened, _ = flatten_list_array(seq)
        trace_indices = pc.list_parent_indices(seq)
        return flattened, trace_indices


class SequenceUnpackWithIndexConfig(BaseDataAugmenterConfig):
    """Configuration class for the :class:`SequenceUnpackWithIndex` augmenter."""


class SequenceUnpackWithIndex(BaseDataAugmenter[SequenceUnpackWithIndexConfig]):
    """Augmenter to unpack a sequence and compute trace indices."""

    @process_mode(batched=True, backend="arrow")
    def process(
        self, ctx: RunContext, seq: Sequence[T]
    ) -> tuple[FlatValuesWithIndex[T], TraceIndexList]:
        """Unpack a sequence and compute trace indices.

        Args:
            ctx (RunContext): Context object containing runtime information.
            seq (Sequence[T]): Input sequence to be flattened.

        Returns:
            tuple[FlatSequenceWithIndex[T], TraceIndexList]: A tuple containing the flattened
            sequence with indices and the computed trace indices.
        """
        # flatten the sequence and compute the trace indices
        flattened, _ = flatten_list_array(seq)
        trace_indices = pc.list_parent_indices(seq)
        # TODO: not sure how expensive this operation is
        #       could use StructArray.from_arrays instead but chunked
        #       array inputs must be handled manually then
        output = pa.table(
            {"value": flattened, "index": trace_indices}, schema=ctx.output_type.arrow_schema
        ).to_struct_array()

        return output, trace_indices.to_numpy()


class SequencePackConfig(BaseDataAugmenterConfig):
    """Configuration class for the :class:`SequencePackUnpacked` augmenter."""

    original_partition: None | PartitionId = None
    """The partition of unpack node in case the values come from a sequence unpack operation."""


class SequencePack(BaseDataAugmenter[SequencePackConfig]):
    """Augmenter to reconstruct a sequence from elements and trace indices."""

    def infer_output_partition(self, ctx: RunContext, partition: PartitionId) -> PartitionId:
        """Infer the output partition of the pack node.

        If the :code:`original_partition` configuration value is set, the output partition
        of the node is explicitly set to the :code:`original_partition`. Otherwise, the output
        partition is determined by the default logic in the parent class.

        Args:
            ctx (RunContext): Execution context for the node.
            partition (PartitionId): The ID of the input partition, i.e. the partition that the
                node is assigned to.

        Returns:
            PartitionId: The output partition ID, corresponding to the node ID of the augmenter.
        """
        return (
            self.config.original_partition
            if self.config.original_partition is not None
            else super(SequencePack, self).infer_output_partition(ctx, partition)
        )

    @process_mode(batched=True, backend="arrow")
    def process(
        self, ctx: RunContext, values: T, trace_index: Int32
    ) -> tuple[Sequence[T], TraceIndexList]:
        """Reconstruct a sequence from flattened elements and trace indices.

        Args:
            ctx (RunContext): Context object containing runtime information.
            values (T): The values to pack into a sequence.
            trace_index (Int32): Indices tracing the original sequence structure.

        Returns:
            tuple[Sequence[T], TraceIndexList]: A tuple containing the reconstructed sequence
            and the offsets for the sequence elements.
        """
        values = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
        trace_index = (
            trace_index.combine_chunks()
            if isinstance(trace_index, pa.ChunkedArray)
            else trace_index
        )
        # compute the offsets where to cut off the array
        offsets = pa.concat_arrays(
            [
                pa.array([0], pa.uint64()),
                pc.indices_nonzero(pc.pairwise_diff(trace_index)),
                pa.array([len(trace_index)], pa.uint64()),
            ]
        )
        # unflatten the sequence
        return unflatten_list_array(values, offsets), offsets[:-1].to_numpy()


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


@SequenceFeature.register_method("pack")
def pack_sequence(values: T, trace_index: Int32, node: None | Reference = None) -> Sequence[T]:
    """Reconstruct a sequence from values and trace indices.

    This method leverages the :class:`SequencePack` augmenter to rebuild a sequence from its
    flattened components, ensuring the sequence is packed with its associated trace indices.
    Additionally, it ensures that the sequence is assigned to the appropriate partition based
    on the node performing the flattening operation.

    Args:
        values (T): The flattened sequence values to be packed.
        trace_index (Int32): The trace indices associated with the values, mapping them to their
            original structure.
        node (None | Reference): A reference to the node performing the flattening operation,
            used to infer the partition for the packing operation.

    Returns:
        Sequence[T]: The packed sequence, reconstructed from the flattened values and trace indices,
        assigned to the appropriate partition for the unflattening operation.
    """
    # get the partition of the node that performs the flattening operation
    # and use it as the target partition of the unflattening operation
    node_partition = (
        node._graph.nodes[node._node_id][DataFlowGraph.NodeAttribute.PARTITION]
        if node is not None
        else None
    )
    return SequencePack(original_partition=node_partition).call(values, trace_index)


SequenceFeature.register_method("sum")(SequenceSum().call)
SequenceFeature.register_method("length")(Length().call)
SequenceFeature.register_method("unpack")(SequenceUnpack().call)
SequenceFeature.register_method("unpack_with_index")(SequenceUnpackWithIndex().call)
