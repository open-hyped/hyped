"""This module defines a collection of data processors that implement common sequence operations.

Each processor is designed to handle a specific sequence transformation or query, such as
calculating lengths, slicing or aggregation operations. These processors are intended for
use in data processing pipelines, where they can be applied in a batched and efficient
manner using Apache Arrow as the backend.

These processors are registered as methods on the :class:`SequenceFeature` class, allowing them
to be applied directly to sequence features.
"""

from typing import Annotated, Any, Generic, TypeVar, overload
from uuid import UUID, uuid5

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from hyped.common._pyarrow import flatten_list_array, unflatten_list_array
from hyped.core.typing import PartitionId

from ..features.dtypes import UNDEFINED_SEQUENCE_LENGTH, Int64Type, MappingType, SequenceType
from ..features.features import (
    Feature,
    Int32Feature,
    IntFeature,
    SequenceFeature,
    build_feature_from_reference,
)
from ..features.reference import ConcreteReference
from ..features.validators import FeatureResolver, Len, MatchFeatures
from ..graph import DataFlowGraph
from ..nodes.augmentor import BaseDataAugmentor, BaseDataAugmentorConfig
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


NumericType = TypeVar("NumericType", bound=Int | Float | UInt)


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


class SequencePadConfig(BaseDataProcessorConfig):
    """Configuration class for the :class:`SequencePad` processor."""

    length: None | int = None
    """The target length to pad each sequence to.

    If :code:`None`, the processor pads sequences to match the length of the longest
    sequence in the current batch.
    """


class SequencePad(BaseDataProcessor[SequencePadConfig]):
    """Processor to pad sequences to a specified length.

    This processor pads input sequences with a specified fill value to either a
    user-defined target length or, if not provided, the length of the longest
    sequence in the current batch.
    """

    @process_mode(batched=True, backend="python")
    def process(
        self, ctx: RunContext, seq: Sequence[ItemType], fill_value: ItemType
    ) -> Annotated[
        Sequence[ItemType],
        FeatureResolver(
            lambda c, i, _: (
                Sequence[ItemType]
                if c.length is None
                else Annotated[Sequence[ItemType], Len(c.length)]
            )
        ),
    ]:
        """Pad sequences to a uniform length with a fill value.

        Args:
            ctx (RunContext): Context object containing runtime information.
            seq (Sequence[ItemType]): A batch of sequences to be padded.
            fill_value (ItemType): The value to use for padding.

        Returns:
            Sequence[ItemType]: A batch of padded sequences. Each sequence will have
            the same length, either matching the user-specified target length or
            the longest sequence in the batch if no target length is specified.
        """
        # get length to pad each sequence to
        length = self.config.length or max(map(len, seq))
        return [s + [v] * (length - len(s)) for s, v in zip(seq, fill_value, strict=True)]


class SequenceGetItemConfig(BaseDataProcessorConfig):
    """Configuration class for the :class:`SequenceGetItem` processor."""


class SequenceGetItem(BaseDataProcessor[SequenceGetItemConfig]):
    """Processor to retrieve an item from a sequence based on a specific index.

    This processor extracts a single element from a sequence at the position specified
    by the :code:`index` argument.
    """

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, seq: Sequence[ItemType], index: Int) -> ItemType:
        """Retrieve an item from a sequence.

        Args:
            ctx (RunContext): Context object containing runtime information.
            seq (Sequence[ItemType]): Input sequence from which to extract the item.
            index (Int): Index to get.

        Returns:
            ItemType: The item at the specified index in the sequence.
        """
        seq = seq.combine_chunks() if isinstance(seq, pa.ChunkedArray) else seq
        index = index.combine_chunks() if isinstance(index, pa.ChunkedArray) else index

        # flatten
        _, seq_offsets = flatten_list_array(seq)
        seq_offsets = (
            np.arange(len(seq) + 1) * seq_offsets if isinstance(seq_offsets, int) else seq_offsets
        )
        # convert negative indices to positives
        index = pc.if_else(pc.less(index, 0), pc.add(index, seq.value_lengths()), index)
        # convert indices to indices in flattened sequence array
        flat_indices = pc.add(index, seq_offsets[:-1])
        # collect items
        return pc.take(seq.flatten(), flat_indices)


class SequenceGetItemsConfig(BaseDataProcessorConfig):
    """Configuration class for the :class:`SequenceGetItems` processor."""


L = Len()


class SequenceGetItems(BaseDataProcessor[SequenceGetItemsConfig]):
    """Processor to retrieve an item from a sequence based on a specific index.

    This processor extracts a subsequence from a sequence at the positions specified
    by the :code:`index` argument.
    """

    @process_mode(batched=True, backend="arrow")
    def process(
        self, ctx: RunContext, seq: Sequence[ItemType], index: Annotated[Sequence[Int], L]
    ) -> Annotated[Sequence[ItemType], L]:
        """Retrieve an item from a sequence.

        Args:
            ctx (RunContext): Context object containing runtime information.
            seq (Sequence[ItemType]): Input sequence from which to extract the item.
            index (Sequence[Int]): Sequence of indices to get.

        Returns:
            Sequence[ItemType]: The items at the specified indices in the sequence.
        """
        seq = seq.combine_chunks() if isinstance(seq, pa.ChunkedArray) else seq
        index = index.combine_chunks() if isinstance(index, pa.ChunkedArray) else index
        # flatten
        _, seq_offsets = flatten_list_array(seq)
        seq_offsets = (
            np.arange(len(seq) + 1) * seq_offsets if isinstance(seq_offsets, int) else seq_offsets
        )
        # print(seq_offsets)
        index_flatten, index_offsets = flatten_list_array(index)
        # convert negative indices to positives
        lengths_flatten = np.array(seq.value_lengths()).repeat(index.value_lengths())
        index_flatten = pc.if_else(
            pc.less(index_flatten, 0), pc.add(index_flatten, lengths_flatten), index_flatten
        )
        # convert flattened indices to indices in flattened sequence array
        flat_indices = pc.add(
            index_flatten, np.array(seq_offsets[:-1]).repeat(index.value_lengths())
        )
        items_flatten = pc.take(seq.flatten(), flat_indices)
        # return unflattened items
        return unflatten_list_array(items_flatten, index_offsets)


class SequenceGetSliceConfig(BaseDataAugmentorConfig):
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


class SequenceZipConfig(BaseDataProcessorConfig):
    """Configuration for SequenceZip operation."""


SequenceZipValType = TypeVar("SequenceZipValType", bound=Feature)
SequenceZipLength = Len()


class SequenceZip(BaseDataProcessor[SequenceZipConfig]):
    """Data Processor for zipping sequences."""

    @process_mode(batched=True, backend="arrow")
    def process(
        self,
        ctx: RunContext,
        **seqs: Annotated[
            Sequence[Annotated[SequenceZipValType, MatchFeatures()]], SequenceZipLength
        ],
    ) -> Annotated[
        Sequence[
            Annotated[
                Sequence[SequenceZipValType],
                FeatureResolver(
                    lambda c, i, _: Annotated[Sequence[SequenceZipValType], Len(len(i))]
                ),
            ]
        ],
        SequenceZipLength,
    ]:
        """Zip multiple sequences.

        Args:
            ctx (RunContext): Context object containing runtime information.
            **seqs (Sequence[SequenceType]): Input sequences to zip. Zipped sequences are ordered
                by their keys, which are converted to integers.

        Returns:
            Sequence[Sequence[SequenceType]]: The zipped sequences.
        """
        # sort the sequences
        seqs_sorted = [seqs[key] for key in sorted(seqs.keys(), key=int)]
        # flatten the sequences along the batch axis
        # flat_seqs: [seq_1_flattened, seq_1_flattened, ...]
        flat_seqs, offsets = zip(*[flatten_list_array(seq) for seq in seqs_sorted], strict=True)
        # check that all sequences have the same lengths
        assert all(offset == offsets[0] for offset in offsets), "Sequences must have same lengths!"
        # concatenate the sequences
        flat_concat = pa.chunked_array(flat_seqs).combine_chunks()
        # get number of sequences and flattened sequence length
        num_seqs = len(seqs_sorted)
        flattened_seq_len = len(flat_seqs[0])
        # create zip indices
        # zip_indices: [[0, 5], [1, 6], [2, 7], ...]
        zip_indices = np.add.outer(
            np.arange(flattened_seq_len), flattened_seq_len * np.arange(num_seqs)
        )
        # interleave the flattened sequences with the zip indices
        flat_interleave = pc.take(flat_concat, zip_indices.flatten())
        # create an list-array of list-array
        flat_zipped = pa.FixedSizeListArray.from_arrays(
            flat_interleave, type=ctx.output_dtype.value_type.arrow_type
        )
        # unflatten to get back the batch axis
        return unflatten_list_array(flat_zipped, offsets[0])


class SequenceZipMappingConfig(BaseDataProcessorConfig):
    """Configuration for SequenceZip operation mapping mixed DTypes."""


MixedSequenceType = TypeVar("MixedSequenceType", bound=Feature)


class SequenceZipMapping(BaseDataProcessor[SequenceZipMappingConfig]):
    """Data Processor for zipping mapping of sequences to sequence of mapping."""

    @process_mode(batched=True, backend="arrow")
    def process(
        self, ctx: RunContext, **seqs: Annotated[Sequence[MixedSequenceType], SequenceZipLength]
    ) -> Annotated[
        Sequence[
            Annotated[
                Mapping,
                FeatureResolver(
                    lambda c, i, _: MappingType.construct(
                        fields={k: f.dtype.value_type for k, f in i.items()}
                    )
                ),
            ]
        ],
        SequenceZipLength,
    ]:
        """Zip a dict-of-sequences into a sequence-of-dicts.

        Args:
            ctx (RunContext): Context object containing runtime information.
            **seqs (Sequence[MixedSequenceType]): Input sequences to zip. The output mapping
                depends on the keys of this dictionary.

        Returns:
            Sequence[Mapping]: The zipped sequences.
        """
        flat_seqs = {}
        offsets = None
        # flatten all sequences
        for key, seq in seqs.items():
            flat_seqs[key], seq_offsets = flatten_list_array(seq)
            # make sure offsets match
            assert pc.all(offsets == seq_offsets), "Trying to zip sequences of different lengths"
            offsets = seq_offsets
        # zip flattened sequences
        zipped_flat_seqs = pa.table(flat_seqs, schema=ctx.output_dtype.value_type.arrow_schema)
        zipped_flat_seqs = zipped_flat_seqs.to_struct_array().combine_chunks()
        # unflatten
        return unflatten_list_array(zipped_flat_seqs, offsets)


class SequenceUnpackConfig(BaseDataAugmentorConfig):
    """Configuration class for the :class:`SequenceUnpack` augmentor."""


class SequenceUnpack(BaseDataAugmentor[SequenceUnpackConfig]):
    """Augmentor to unpack a sequence."""

    def infer_output_partition(self, ctx: RunContext, partition: PartitionId) -> PartitionId:
        """Determine the output partition of the unpack operation.

        For fixed-length sequences the output partition is deterministically computed
        from the node partition and the length of the sequence. This allows to combine
        elements from different unpacked sequences.

        Args:
            ctx (RunContext): Execution context for the node.
            partition (PartitionId): The ID of the input partition, i.e. the partition that the
                node is assigned to.

        Returns:
            PartitionId: The output partition ID, corresponding to the node ID of the augmentor.
        """
        length: int = ctx.input_dtype["seq"].length
        if length != UNDEFINED_SEQUENCE_LENGTH:
            return str(uuid5(UUID(partition), length.to_bytes(4, byteorder="big").decode()))

        return super(SequenceUnpack, self).infer_output_partition(ctx, partition)

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


class SequenceUnpackWithIndexConfig(BaseDataAugmentorConfig):
    """Configuration class for the :class:`SequenceUnpackWithIndex` augmentor."""


class SequenceUnpackWithIndex(BaseDataAugmentor[SequenceUnpackWithIndexConfig]):
    """Augmentor to unpack a sequence and compute trace indices."""

    def infer_output_partition(self, ctx: RunContext, partition: PartitionId) -> PartitionId:
        """Determine the output partition of the unpack operation.

        For fixed-length sequences the output partition is deterministically computed
        from the node partition and the length of the sequence. This allows to combine
        elements from different unpacked sequences.

        Args:
            ctx (RunContext): Execution context for the node.
            partition (PartitionId): The ID of the input partition, i.e. the partition that the
                node is assigned to.

        Returns:
            PartitionId: The output partition ID, corresponding to the node ID of the augmentor.
        """
        length: int = ctx.input_dtype["seq"].length
        if length != UNDEFINED_SEQUENCE_LENGTH:
            return str(uuid5(UUID(partition), length.to_bytes(4, byteorder="big").decode()))

        return super(SequenceUnpackWithIndex, self).infer_output_partition(ctx, partition)

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

        if len(flattened) > 0:
            trace_indices = pc.list_parent_indices(seq)
            # zip values with trace indices
            output = pa.table(
                {"value": flattened, "index": trace_indices}, schema=ctx.output_dtype.arrow_schema
            )
            return output.to_struct_array(), trace_indices.to_pylist()

        else:
            output = pa.chunked_array([], type=ctx.output_dtype.arrow_type)
            return output, []


class SequencePackConfig(BaseDataAugmentorConfig):
    """Configuration class for the :class:`SequencePackUnpacked` augmentor."""

    original_partition: None | PartitionId = None
    """The partition of unpack node in case the values come from a sequence unpack operation."""

    original_length: None | int = None
    """The length of the sequence before the unpack operation if defined."""


class SequencePack(BaseDataAugmentor[SequencePackConfig]):
    """Augmentor to reconstruct a sequence from elements and trace indices."""

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
            PartitionId: The output partition ID, corresponding to the node ID of the augmentor.
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
        diffs = np.diff(trace_index.to_numpy(), prepend=0)
        offsets = np.repeat(np.arange(len(diffs)), diffs)
        offsets = np.concatenate(
            (
                np.asarray([0], dtype=offsets.dtype),
                offsets,
                np.asarray(
                    [len(diffs)]
                    * (
                        1
                        if ctx.target_batch_size is None
                        else (ctx.target_batch_size - len(offsets))
                    ),
                    dtype=offsets.dtype,
                ),
            ),
            dtype=offsets.dtype,
        )
        # pack sequence
        packed_sequence = unflatten_list_array(
            array=values,
            offsets=(
                pa.array(offsets)
                if self.config.original_length is None
                else self.config.original_length
            ),
        )
        # compute the trace index
        out_trace_index = offsets[:-1]
        out_trace_index[offsets[:-1] == offsets[1:]] = -1
        # return the packed sequence and the trace indices
        return packed_sequence, out_trace_index


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


@SequenceFeature.register_method("pad")
def sequence_pad(seq: Sequence[ItemType], fill_value: ItemType, length: None | int) -> ItemType:
    """Pad the sequence to a specified length with a given fill value.

    If :code:`length` is provided, the sequence is padded to the specified length.
    If :code:`length` is :code:`None`, the sequence is padded to match the length
    of the longest sequence in the current batch.

    Args:
        seq (Sequence[ItemType]): The sequence to pad.
        fill_value (ItemType): The value to use for padding.
        length (None | int): The desired length to pad the sequence to.
            Defaults to `None`, in which case the longest sequence in the batch is used.

    Returns:
        SequenceFeature[T]: A new :class:`SequenceFeature` instance containing the
        padded sequence.
    """
    return SequencePad(length=length).call(seq, fill_value)


@SequenceFeature.register_method("__getitem__")
def sequence_get_item(
    sequence: Sequence[ItemType], index: int | slice | list | Int | Sequence[Int]
) -> ItemType:
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
    if isinstance(index, (int, IntFeature)):
        return SequenceGetItem().call(seq=sequence, index=index)

    elif isinstance(index, SequenceFeature):
        return SequenceGetItems().call(seq=sequence, index=index)

    # TODO: Remove this once `builder.compute` fixes the auto-adding of const sequences
    elif isinstance(index, list):
        ref = sequence.ref._builder.const(index, SequenceType(Int64Type, len(index)))
        index_feature = build_feature_from_reference(ref)
        return SequenceGetItems().call(seq=sequence, index=index_feature)

    elif isinstance(index, slice):
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

    else:
        raise ValueError(f"Index type not supported, got {index}")


@overload
def zip_(*args: Sequence) -> Sequence[Sequence]:
    ...


@overload
def zip_(**kwargs: Sequence) -> Sequence[Mapping]:
    ...


def zip_(*args: Sequence, **kwargs: Sequence) -> Sequence[Sequence] | Sequence[Mapping]:
    """Zip multiple sequences together.

    Args:
        *args (Sequence): Sequences to zip together. Must all have the same Value type.
        **kwargs (Sequence): Sequences to zip together. Can have mixed value types.

    Returns:
        Sequence[Sequence] | Sequence[Mapping]: The zipped sequences. If the input is
            a list of sequences the output will be a :code:`Sequence[Sequence]` too. If the input
            is a dict of mixed-type sequences, the output will be a :code:`Sequence[Mapping]`.
    """
    if len(args) > 0 and len(kwargs) > 0:
        raise ValueError("Must specify either sequences as *args or **kwargs, not both!")

    if len(args):
        proc_kwargs = {str(i): args[i] for i in range(len(args))}
        try:
            return SequenceZip().call(**proc_kwargs)
        except TypeError as e:
            raise TypeError(
                "The value features of the sequences passed to `zip_` don't align! "
                f"Got {[seq.dtype.value_type.arrow_type for seq in args]}. If you want to "
                "zip sequences of mixed value types, pass kwargs to the zip_ function "
                "instead, to create a sequence-of-mapping."
            ) from e
    else:
        return SequenceZipMapping().call(**kwargs)


@SequenceFeature.register_method("pack")
def pack_sequence(
    values: ItemType, trace_index: Int32, node: None | ConcreteReference = None
) -> Sequence[ItemType]:
    """Reconstruct a sequence from values and trace indices.

    This method leverages the :class:`SequencePack` augmentor to rebuild a sequence from its
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
        node._graph.nodes[node._node_id][DataFlowGraph.NodeAttribute.OUT_PARTITION]
        if node is not None
        else None
    )
    length = len(node.get_dtype()) if node is not None else None
    length = length if length != UNDEFINED_SEQUENCE_LENGTH else None

    pack = SequencePack(original_partition=partition, original_length=length)
    return pack.call(values, trace_index)


SequenceFeature.register_method("sum")(SequenceSum().call)
SequenceFeature.register_method("length")(SequenceLength().call)
SequenceFeature.register_method("unpack")(SequenceUnpack().call)
SequenceFeature.register_method("unpack_with_index")(SequenceUnpackWithIndex().call)
