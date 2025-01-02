"""This module defines a collection of data processors that implement common sequence operations.

Each processor is designed to handle a specific sequence transformation or query, such as
calculating lengths, slicing or aggregation operations. These processors are intended for
use in data processing pipelines, where they can be applied in a batched and efficient
manner using Apache Arrow as the backend.

These processors are registered as methods on the :class:`SequenceFeature` class, allowing them
to be applied directly to sequence features.
"""

from typing import Annotated, Any, Generic, TypeVar

import pyarrow as pa
import pyarrow.compute as pc

from hyped.common._pyarrow import flatten_list_array, unflatten_list_array
from hyped.core.typing import PartitionId

from ..features.dtypes import UNDEFINED_SEQUENCE_LENGTH
from ..features.features import Int32Feature, SequenceFeature
from ..features.reference import ConcreteReference
from ..features.validators import FeatureResolver, Len
from ..graph import DataFlowGraph
from ..nodes.augmenter import BaseDataAugmenter, BaseDataAugmenterConfig
from ..nodes.base import RunContext, process_mode
from ..nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from ..typing import Float, Int, Int32, Mapping, Sequence, TraceIndexList, UInt


class SequenceLengthConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`SequenceLength` processor."""


class SequenceLength(BaseDataProcessor[SequenceLengthConfig]):
    """Data processor for computing the length of sequences."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, seq: Sequence) -> Int32Feature:
        """Computes the length of an input sequence.

        Args:
            ctx (RunContext): The execution context.
            seq (Sequence): The sequence to compute the length for.

        Returns:
            IntFeature: The lengths of the input sequences.
        """
        return pc.list_value_length(seq)


NumericType = TypeVar("T", bound=Int | Float | UInt)


class SequenceMinConfig(BaseDataProcessorConfig):
    """Configuration class for the SequenceMin processor."""

    default: None | int | float = None
    """The default value to return if the sequence is empty."""


class SequenceMin(BaseDataProcessor[SequenceMinConfig]):
    """Processor to compute the minimum value of a numeric sequence."""

    @process_mode(batched=False, backend="python")
    def process(self, ctx: RunContext, seq: Sequence[NumericType]) -> NumericType:
        """Compute the minimum value of a sequence.

        Args:
            ctx (RunContext): Context object containing runtime information.
            seq (Sequence[NumericType]): Input sequence of numeric values.

        Returns:
            NumericType: The minimum value in the sequence, or the default value
            specified in the configuration if the sequence is empty.
        """
        return min(seq, default=self.config.default)


class SequenceMaxConfig(BaseDataProcessorConfig):
    """Configuration class for the SequenceMax processor."""

    default: None | int | float = None
    """The default value to return if the sequence is empty."""


class SequenceMax(BaseDataProcessor[SequenceMaxConfig]):
    """Processor to compute the maximum value of a numeric sequence."""

    @process_mode(batched=False, backend="python")
    def process(self, ctx: RunContext, seq: Sequence[NumericType]) -> NumericType:
        """Compute the maximum value of a sequence.

        Args:
            ctx (RunContext): Context object containing runtime information.
            seq (Sequence[NumericType]): Input sequence of numeric values.

        Returns:
            NumericType: The maximum value in the sequence, or the default value
            specified in the configuration if the sequence is empty.
        """
        return max(seq, default=self.config.default)


class SequenceSumConfig(BaseDataProcessorConfig):
    """Configuration class for the SequenceSum processor."""


class SequenceSum(BaseDataProcessor[SequenceSumConfig]):
    """Processor to compute the sum of numeric values in a sequence."""

    @process_mode(batched=False, backend="python")
    def process(self, ctx: RunContext, seq: Sequence[NumericType]) -> NumericType:
        """Compute the sum of a sequence.

        Args:
            ctx (RunContext): Context object containing runtime information.
            seq (Sequence[NumericType]): Input sequence of numeric values.

        Returns:
            NumericType: The sum of the values in the sequence, or the default value
            specified in the configuration if the sequence is empty.
        """
        return sum(seq)


ItemType = TypeVar("ItemType")


class SequenceGetItemConfig(BaseDataAugmenterConfig):
    """Configuration class for the :class:`SequenceGetItem` processor."""

    index: int
    """The index of the item to retrieve from the sequence."""


class SequenceGetItem(BaseDataProcessor[SequenceGetItemConfig]):
    """Processor to retrieve an item from a sequence based on a specific index.

    This processor extracts a single element from a sequence at the position specified
    by the :code:`index` attribute in its configuration.
    """

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, seq: Sequence[ItemType]) -> ItemType:
        """Retrieve an item from a sequence.

        Args:
            ctx (RunContext): Context object containing runtime information.
            seq (Sequence[ItemType]): Input sequence from which to extract the item.

        Returns:
            ItemType: The item at the specified index in the sequence.
        """
        return pc.list_element(seq, self.config.index)


class SequenceGetSliceConfig(BaseDataAugmenterConfig):
    """Configuration class for the :class:`SequenceGetSlice` processor."""

    start: int
    """The starting index of the slice (inclusive)."""

    stop: None | int
    """The ending index of the slice (exclusive)."""

    step: int
    """The step size for the slice."""


class SequenceGetSlice(BaseDataProcessor[SequenceGetSliceConfig]):
    """Processor to extract a slice from a sequence.

    This processor extracts a sub-sequence from a given sequence using
    slicing parameters defined in its configuration.
    """

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, seq: Sequence[ItemType]) -> Sequence[ItemType]:
        """Extract a slice from a sequence.

        Args:
            ctx (RunContext): Context object containing runtime information.
            seq (Sequence[ItemType]): Input sequence from which to extract the slice.

        Returns:
            Sequence[ItemType]: The sub-sequence defined by the start, stop, and step indices.
        """
        return pc.list_slice(seq, self.config.start, self.config.stop, self.config.step)


class SequenceValueWithIndex(Mapping, Generic[ItemType]):
    """Represents a sequence value and the origin batch index."""

    value: ItemType
    """The sequence value."""

    index: Int32Feature
    """The index of the batch containing the sequence that the value originates from."""


class SequenceUnpackConfig(BaseDataAugmenterConfig):
    """Configuration class for the :class:`SequenceUnpack` augmenter."""


class SequenceUnpack(BaseDataAugmenter[SequenceUnpackConfig]):
    """Augmenter to unpack a sequence."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, seq: Sequence[ItemType]) -> tuple[ItemType, TraceIndexList]:
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
        return flattened, trace_indices.to_pylist()


class SequenceUnpackWithIndexConfig(BaseDataAugmenterConfig):
    """Configuration class for the :class:`SequenceUnpackWithIndex` augmenter."""


class SequenceUnpackWithIndex(BaseDataAugmenter[SequenceUnpackWithIndexConfig]):
    """Augmenter to unpack a sequence and compute trace indices."""

    @process_mode(batched=True, backend="arrow")
    def process(
        self, ctx: RunContext, seq: Sequence[ItemType]
    ) -> tuple[SequenceValueWithIndex[ItemType], TraceIndexList]:
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

        return output, trace_indices.to_pylist()


class SequencePackConfig(BaseDataAugmenterConfig):
    """Configuration class for the :class:`SequencePackUnpacked` augmenter."""

    original_partition: None | PartitionId = None
    """The partition of unpack node in case the values come from a sequence unpack operation."""

    original_length: None | int = None
    """The length of the sequence before the unpack operation if defined."""


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
        self, ctx: RunContext, values: ItemType, trace_index: Int32
    ) -> tuple[
        Annotated[
            Sequence[ItemType],
            FeatureResolver(
                lambda c, _, s: (
                    Sequence[ItemType]
                    if c.original_length is None
                    else Annotated[Sequence[ItemType], Len(c.original_length)]
                )
            ),
        ],
        TraceIndexList,
    ]:
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
        cutoffs = offsets if self.config.original_length is None else self.config.original_length
        return unflatten_list_array(values, cutoffs), offsets[:-1].to_pylist()


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


@SequenceFeature.register_method("__getitem__")
def sequence_get_item(sequence: Sequence[ItemType], index: int | slice) -> ItemType:
    """Retrieve an item or a subsequence from a sequence using an integer index or a slice.

    This method uses the :class:`SequenceGetItem` and :class:`SequenceGetSlice` processors
    to handle both single-item retrieval and slicing. It supports sequences with defined
    or undefined lengths, but imposes restrictions for negative indices or slices when the
    sequence length is unknown.

    Args:
        sequence (Sequence[ItemType]): The input sequence feature.
        index (int | slice): The index or slice used for retrieval. Negative indices
            are supported only for sequences with defined lengths.

    Returns:
        ItemType: The retrieved item or subsequence, based on the provided index.

    Raises:
        RuntimeError: If negative indices or slices are used with sequences of
            undefined length.
    """
    if isinstance(index, int):
        if (index < 0) and len(sequence.dtype) != UNDEFINED_SEQUENCE_LENGTH:
            index = len(sequence.dtype) + index

        elif index < 0:
            raise RuntimeError(
                "Negative indices are not allowed for lists with unknown lengths, "
                f"got index {index}."
            )

        return SequenceGetItem(index=index).call(sequence)

    if isinstance(index, slice):
        if len(sequence.dtype) != UNDEFINED_SEQUENCE_LENGTH:
            start, stop, step = index.indices(len(sequence.dtype))

        else:
            # prepare the slice, cannot use .indices here because we don't know the length
            stop = index.stop
            start = index.start if index.start is not None else 0
            step = index.step if index.step is not None else 1

            if (start < 0) or ((stop is not None) and (stop < 0)):
                # not supported for lists of unkown length
                raise RuntimeError(
                    "Negative stop values are not allowed for lists with "
                    f"unknown lengths, got slice ({start}, {stop}, {step})."
                )

        return SequenceGetSlice(start=start, stop=stop, step=step).call(sequence)

    raise NotImplementedError(f"Index type not supported, got {index}")


@SequenceFeature.register_method("pack")
def pack_sequence(
    values: ItemType, trace_index: Int32, node: None | ConcreteReference = None
) -> Sequence[ItemType]:
    """Reconstruct a sequence from values and trace indices.

    This method leverages the :class:`SequencePack` augmenter to rebuild a sequence from its
    flattened components, ensuring the sequence is packed with its associated trace indices.
    Additionally, it ensures that the sequence is assigned to the appropriate partition based
    on the node performing the flattening operation.

    Args:
        values (T): The flattened sequence values to be packed.
        trace_index (Int32): The trace indices associated with the values, mapping them to their
            original structure.
        node (None | ConcreteReference): A reference to the node performing the flattening
            operation, used to infer the partition for the packing operation.

    Returns:
        Sequence[T]: The packed sequence, reconstructed from the flattened values and trace indices,
        assigned to the appropriate partition for the unflattening operation.
    """
    # get the partition of the node that performs the flattening operation
    # and use it as the target partition of the unflattening operation
    partition = (
        node._graph.nodes[node._node_id][DataFlowGraph.NodeAttribute.PARTITION]
        if node is not None
        else None
    )
    length = len(node.get_dtype())
    length = length if length != UNDEFINED_SEQUENCE_LENGTH else None

    pack = SequencePack(original_partition=partition, original_length=length)
    return pack.call(values, trace_index)


SequenceFeature.register_method("sum")(SequenceSum().call)
SequenceFeature.register_method("length")(SequenceLength().call)
SequenceFeature.register_method("unpack")(SequenceUnpack().call)
SequenceFeature.register_method("unpack_with_index")(SequenceUnpackWithIndex().call)
