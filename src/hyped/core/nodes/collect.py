"""This module defines the :class:`CollectNode`.

The :class:`CollectNode` class is a specialized node that gathers and structures data from input
references based on a predefined lookup structure. This structure can be a dictionary, list, tuple,
or individual feature, and the node ensures that the output matches the specified data types as
defined in the data flow graph.
"""

from functools import partial
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..abstract import AbstractDataFlowGraph
from ..features.dtypes import (
    UNDEFINED_SEQUENCE_LENGTH,
    DType,
    MappingType,
    SequenceType,
    cast_dtype,
    common_dtype,
)
from ..features.reference import ConcreteReference
from ..utils import NestedType
from .base import BaseNode, BaseNodeConfig, RunContext


class CollectNodeConfig(BaseNodeConfig):
    """Configuration for CollectNode."""

    lookup: dict | list | tuple | str
    """Nested collect structure.

    The structure has string identifiers matching the input arguments at the
    lowest level.
    """


class CollectNode(BaseNode[CollectNodeConfig]):
    """Node for collecting and structuring inputs within a data flow graph.

    :class:`CollectNode` aggregates data from multiple inputs according to a defined structure
    specified in its configuration. It transforms the collected data into the desired output
    format, supporting nested mappings and sequences.
    """

    def build_output_type(
        self, graph: AbstractDataFlowGraph, inputs: dict[str, ConcreteReference]
    ) -> tuple[DType, dict[str, DType]]:
        """Construct the output data type based on the input structure.

        This method recursively determines the type of each element in the structure specified by
        :code:`lookup`, building complex types like mappings and sequences as needed. It leverages
        input references to map strings in :code:`lookup` to actual data types from the graph.

        Args:
            graph (AbstractDataFlowGraph): The data flow graph containing the input references.
            inputs (dict[str, ConcreteReference]): Dictionary mapping input names to references
                within the graph.

        Returns:
            tuple[DType, dict[str, DType]]: A tuple containing the constructed output type and a
                lookup for inputs that need to be casted to a different data type before
                collection.
        """
        # dictionary mapping inputs to the dtype they need to be casted to
        required_casts: dict[str, DType] = {}

        def _cast(obj: NestedType[str], src_dtype: DType, tgt_dtype: DType) -> None:
            # trivial case: no type casting required
            if src_dtype == tgt_dtype:
                return

            assert isinstance(obj, (str, dict, list, tuple)), f"Unexpected type: {obj}"

            if isinstance(obj, str):
                required_casts[obj] = cast_dtype(src_dtype, tgt_dtype)

            elif isinstance(obj, dict):
                assert isinstance(src_dtype, MappingType) and isinstance(tgt_dtype, MappingType)
                for key, item in obj.items():
                    _cast(item, src_dtype[key], tgt_dtype[key])

            elif isinstance(obj, (list, tuple)):
                assert isinstance(src_dtype, SequenceType) and isinstance(tgt_dtype, SequenceType)
                for item in obj:
                    _cast(item, src_dtype.value_type, tgt_dtype.value_type)

        def _build_type(obj: NestedType[str]):
            if isinstance(obj, str):
                # get the data type from the graph
                return graph.get_dtype_from_reference(inputs[obj])

            elif isinstance(obj, dict):
                return MappingType.construct({key: _build_type(item) for key, item in obj.items()})

            elif isinstance(obj, (list, tuple)):
                if len(obj) == 0:
                    raise RuntimeError("Empty sequences are not supported in the lookup structure.")

                dtype, *others = map(_build_type, obj)
                # check the data types of all sequence items
                if any(dtype != other for other in others):
                    # cast the values in the sequence to a common type
                    target_dtype = common_dtype(dtype, *others)
                    for o, t in zip(obj, (dtype, *others), strict=True):
                        _cast(o, t, target_dtype)
                    # update data type
                    dtype = target_dtype

                # build the sequence type
                return SequenceType(value_type=dtype, length=len(obj))

            else:  # pragma: not covered
                raise TypeError(
                    f"Unsupported type encountered in lookup structure, got {type(obj).__name__}."
                )

        return _build_type(self.config.lookup), required_casts

    def collect(self, ctx: RunContext, inputs: dict[str, pa.Array]) -> pa.Array:
        """Collect and arrange input data according to the node’s structure.

        Recursively gathers data from the input arrays, organizing it into a nested format
        as specified in :code:`lookup`. The method supports nested mappings and sequences,
        building them using PyArrow structures like :class:`StructArray` and
        :class:`FixedSizeListArray`.

        Args:
            ctx (RunContext): The execution context for the current operation, including output
                type.
            inputs (dict[str, pa.Array]): Input data arrays indexed by the input names specified
                in :code:`lookup`.

        Returns:
            pa.Array: A single PyArrow array that represents the collected data, structured
            according to :code:`lookup`.

        Raises:
            TypeError: If the structure or types in :code:`lookup` are not compatible with the
                inputs.
        """

        def _collect(struct: NestedType[str], dtype: DType) -> pa.Array:
            if isinstance(struct, str):
                # get the referenced input array
                return inputs[struct]

            elif isinstance(struct, dict) and isinstance(dtype, MappingType):
                # collect all field values
                field_values = [
                    _collect(struct[field_name], field_dtype)
                    for field_name, field_dtype in dtype.fields
                ]
                field_dtypes = [
                    (field_name, field_dtype.arrow_type) for field_name, field_dtype in dtype.fields
                ]
                # pack values in a struct array
                return pa.StructArray.from_arrays(
                    [
                        array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array
                        for array in field_values
                    ],
                    fields=field_dtypes,
                )

            elif isinstance(struct, (list, tuple)) and isinstance(dtype, SequenceType):
                struct = list(map(partial(_collect, dtype=dtype.value_type), struct))
                # get dimensions
                num_arrays, array_length = len(struct[0]), len(struct)
                assert array_length == dtype.length != UNDEFINED_SEQUENCE_LENGTH, (
                    f"Mismatch between sequence length ({array_length}) and expected "
                    f"length ({dtype.length})."
                )
                # compute the zip reordering indices
                zip_indices = np.add.outer(
                    np.arange(num_arrays), num_arrays * np.arange(array_length)
                )
                # concatenate the arrays and reorder to match the zip view
                flat_struct = pa.chunked_array(
                    struct, type=dtype.value_type.arrow_type
                ).combine_chunks()
                flat_struct = pc.take(flat_struct, zip_indices.flatten())
                # unflatten
                return pa.FixedSizeListArray.from_arrays(flat_struct, type=dtype.arrow_type)

            else:  # pragma: not covered
                raise TypeError(
                    "Unsupported structure or type encountered in collect: "
                    f"{type(struct).__name__} with dtype {dtype}."
                )

        return _collect(self.config.lookup, ctx.output_type)

    @property
    def signature(self) -> Any:  # pragma: not covered
        """Raises an error, as signature is not supported for this node."""
        raise EnvironmentError("The `signature` property is not available for collect nodes.")

    def call(self, *args: Any, **kwargs: Any) -> Any:  # pragma: not covered
        """Raises an error, as direct calls are not supported for this node."""
        raise EnvironmentError("The `call` method is not available for collect nodes.")
