"""Provides sequence operations on feature references.

This module provides a set of operations specifically designed for handling
sequence-based features within a data flow. Sequences are features that 
contain ordered collections of elements (e.g. :code:`Sequence` or :code:`Value("string")`), and
the operations in this module allow for manipulation and querying of these sequences.

Key functionalities include:

- **Length Calculation** (:func:`len_`): Determine the length of a sequence or 
  string-like feature.
- **Element Access and Mutation** (:func:`get_item`, :func:`set_item`): Retrieve or set 
  elements at specific indices within a sequence.
- **Filtering and Searching** (:func:`compress`, :func:`contains`, :func:`count_of`,
  :func:`index_of`): Apply masks to sequences, or search for specific elements within them.
- **Sequence Manipulation** (:func:`chain`, :func:`zip_`): Combine multiple sequences 
  together through concatenation or zipping.
"""
from typing import Any, Iterable

import hyped.nodes._ops as ops
from hyped.common.feature_checks import (
    STRING_LIKE_TYPES,
    check_feature_equals,
    check_feature_is_sequence,
    get_sequence_length,
)
from hyped.core.refs.ref import FeatureRef
from hyped.ops.utils import _check_args, _handle_constant_inputs_for_binary_op, collect


def len_(seq: FeatureRef) -> FeatureRef | int:
    """Determine the length of a feature.

    This function calculates the length of a feature if it is a sequence or a
    string-like type according to the following logic:

    - For sequences, it returns the length as an integer if the length is fixed, or
      as a FeatureRef if the length is dynamic.
    - For string-like features, the length is always assumed to be dynamic, and thus
      results in a FeatureRef instance.

    Args:
        seq (FeatureRef): The feature for which to determine the length.

    Returns:
        FeatureRef | int: The length of the sequence as an integer if fixed, or as a FeatureRef
        if dynamic.

    Raises:
        TypeError: If the feature is of an unexpected type.
    """
    if check_feature_is_sequence(seq.feature_):
        # return constant in case length if fixed and a
        # feature reference to the length feature otherwise
        length = get_sequence_length(seq.feature_)
        return length if length != -1 else (ops.SequenceLength().call(a=seq).result)

    elif check_feature_equals(seq.feature_, STRING_LIKE_TYPES):
        # implement length operation for string-like features
        raise NotImplementedError()

    else:
        # unexpected feature type
        raise TypeError("Unexpected feature type for length operation, " "got `{a.feature_}`.")


def get_item(seq: FeatureRef | Any, index: FeatureRef | Any) -> FeatureRef:
    """Retrieve an item from a sequence feature at a specified index.

    Args:
        seq (FeatureRef | Any): The sequence feature or constant value to retrieve the item from.
        index (FeatureRef | Any): The index at which to retrieve the item. Can also
            be a sequence of indices.

    Returns:
        FeatureRef: The feature representing the item at the specified index.
    """
    # check arguments
    seq, index = _check_args(seq, index)
    # add the getitem processor
    return ops.SequenceGetItem().call(sequence=seq, index=index).result


def set_item(
    seq: FeatureRef | list[Any],
    index: FeatureRef | int | list[int] | slice,
    value: FeatureRef | Any | list[Any],
) -> FeatureRef:
    """Set an item in a sequence feature at a specified index.

    Args:
        seq (FeatureRef): The sequence feature to modify.
        index (FeatureRef | int | list[int] | slice): The index at which
            to set the item. Can also be a sequence of indices or slice.
        value (FeatureRef | Any | list[Any]): The value to set at the
            specified index. Can be a sequence of values in case the
            index is a sequence as well.

    Returns:
        FeatureRef: The feature representing the modified sequence.
    """
    if isinstance(index, slice):
        # TODO: support slices as index
        raise NotImplementedError()
    # check arguments
    seq, index, value = _check_args(seq, index, value)
    # add the setitem processor
    return ops.SequenceSetItem().call(sequence=seq, index=index, value=value).result


@_handle_constant_inputs_for_binary_op
def compress(seq: FeatureRef, mask: FeatureRef) -> FeatureRef:
    """Filter elements from a sequence based on a boolean mask.

    This function applies a boolean mask to a sequence, returning a new sequence
    that includes only the elements where the corresponding mask value is `True`.

    Args:
        seq (FeatureRef): The sequence from which to filter elements.
        mask (FeatureRef): A boolean mask indicating which elements to include
            in the output sequence. The mask should be the same length as `values`.

    Returns:
        FeatureRef: The feature representing the filtered sequence.
    """
    return ops.BooleanIndexing().call(values=seq, mask=mask).indexed_values


def contains(seq: FeatureRef | Any, value: FeatureRef | Any) -> FeatureRef:
    """Check if a sequence or string-like feature contains a specified value.

    Args:
        seq (FeatureRef | Any): The sequence or string-like feature to check.
        value (FeatureRef | Any): The value to check for.

    Returns:
        FeatureRef: A FeatureRef instance representing whether the value
            is contained in the feature.

    Raises:
        TypeError: If the feature is of an unexpected type.
    """
    seq, value = _check_args(seq, value)

    if check_feature_is_sequence(seq.feature_):
        return ops.SequenceContains().call(sequence=seq, value=value).result

    elif check_feature_equals(seq.feature_, STRING_LIKE_TYPES):
        # implement contains operation for string-like features
        raise NotImplementedError()

    else:
        raise TypeError("Unexpected feature type for contains operation, " "got `{obj.feature_}`.")


def count_of(seq: FeatureRef | Any, value: FeatureRef | Any) -> FeatureRef:
    """Count the occurrences of a value in a sequence or string-like feature.

    Args:
        seq (FeatureRef | Any): The sequence or string-like feature to check.
        value (FeatureRef | Any): The value to count occurrences of.

    Returns:
        FeatureRef: A FeatureRef instance representing the count of occurrences.

    Raises:
        TypeError: If the feature is of an unexpected type.
    """
    seq, value = _check_args(seq, value)

    if check_feature_is_sequence(seq.feature_):
        return ops.SequenceCountOf().call(sequence=seq, value=value).result

    elif check_feature_equals(seq.feature_, STRING_LIKE_TYPES):
        # implement contains operation for string-like features
        raise NotImplementedError()

    else:
        raise TypeError("Unexpected feature type for countOf operation, " "got `{obj.feature_}`.")


def index_of(seq: FeatureRef | Any, value: FeatureRef | Any) -> FeatureRef:
    """Find the index of a value in a sequence or string-like feature.

    Args:
        seq (FeatureRef | Any): The sequence or string-like feature to check.
        value (FeatureRef | Any): The value to find the index of.

    Returns:
        FeatureRef: A FeatureRef instance representing the index of the value.

    Raises:
        TypeError: If the feature is of an unexpected type.
    """
    seq, value = _check_args(seq, value)

    if check_feature_is_sequence(seq.feature_):
        return ops.SequenceIndexOf().call(sequence=seq, value=value).result

    elif check_feature_equals(seq.feature_, STRING_LIKE_TYPES):
        # implement contains operation for string-like features
        raise NotImplementedError()

    else:
        raise TypeError("Unexpected feature type for indexOf operation, " "got `{obj.feature_}`.")


def chain(*sequences: FeatureRef) -> FeatureRef:
    """Concatenate sequence features.

    Args:
        *sequences (FeatureRef): Sequence features to chain. Must all have the same Value type.

    Returns:
        FeatureRef: A FeatureRef instance representing the chained sequences.

    Raises:
        TypeError: If the features are of unexpected types.
    """
    sequences = _check_args(*sequences)
    seq_container = collect({str(i): seq for i, seq in enumerate(sequences)})
    # return chained sequence feature
    return ops.SequenceChain().call(sequences=seq_container).result


def zip_(*sequences: FeatureRef) -> FeatureRef:
    """Zip multiple sequences together.

    Args:
        *sequences (FeatureRef): Sequences to zip together. Must all have the same Value type.

    Returns:
        FeatureRef: A FeatureRef instance representing the zipped sequences.

    Raises:
        TypeError: If the features are of unexpected types.
    """
    sequences = _check_args(*sequences)
    seq_container = collect({str(i): seq for i, seq in enumerate(sequences)})
    # zip collected sequences
    return ops.SequenceZip().call(sequences=seq_container).result


def chunk(
    sequences: FeatureRef | Iterable[FeatureRef],
    chunk_size: int,
    stride: None | int = None,
    keep_last: bool = True,
) -> tuple[FeatureRef, ...]:
    """Splits one or more sequences into smaller chunks.

    This function takes one or more sequences (assumed to have the same length) and divides
    them into sub-sequences (chunks) of length :code:`chunk_size`. It can also control whether
    these chunks overlap by specifying a :code:`stride` value, and it can optionally keep or
    discard the final chunk if it is smaller than :code:`chunk_size`.

    Args:
        sequences (FeatureRef | Iterable[FeatureRef]): One or more sequences to be chunked.
            If a single sequence is provided, it will be wrapped in a list. Each sequence
            should be of the same length.
        chunk_size (int): The number of elements in each chunk. This is the size of
            the sub-sequences that will be created.
        stride (None | int, optional): The step size between the start of consecutive chunks.
            If :code:`None`, it defaults to :code:`chunk_size` (non-overlapping chunks). A smaller
            stride value creates overlapping chunks. Defaults to `None`.
        keep_last (bool, optional): Determines whether to keep the last chunk if it contains
            fewer elements than :code:`chunk_size`. If :code:`True`, the last chunk is included
            even if it is smaller than :code:`chunk_size`. If :code:`False`, the incomplete final
            chunk is discarded. Defaults to :code:`True`.

    Returns:
        tuple[FeatureRef, ...]: A tuple of chunked sequences, where each sequence is split
            in parallel with others into chunks based on the `chunk_size` and `stride` values.

    Example:
        .. code-block:: python

            seq1: FeatureRef  # [1, 2, 3, 4, 5, 6]
            seq2: FeatureRef  # [10, 20, 30, 40, 50, 60]
            chunks1, chunks2 = chunk([seq1, seq2], chunk_size=2)

        The resulting chunks will be:

        - chunks1: :code:`(1, 2), (3, 4), (5, 6)`
        - chunks2: :code:`(10, 20), (30, 40), (50, 60)`

        With :code:`stride=1`, the chunks overlap:

        .. code-block:: python

            chunks1: :code:`(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)`
            chunks2: :code:`(10, 20), (20, 30), (30, 40), (40, 50), (50, 60)`
    """

    if isinstance(sequences, FeatureRef):
        sequences = [sequences]

    # create sequence chunker object
    chunker = ops.SequenceChunk(
        chunk_size=chunk_size, stride=stride or chunk_size, keep_last=keep_last
    )
    # collect all sequences into a single feature and apply chunker
    sequences = collect({str(i): s for i, s in enumerate(sequences)})
    chunks = chunker.call(sequences)

    # output each chunk
    return tuple(chunks[k] for k in chunks.feature_.keys())
