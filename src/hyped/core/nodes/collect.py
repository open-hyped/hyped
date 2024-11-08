from functools import partial
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..abstract import AbstractDataFlowGraph
from ..features.reference import Reference
from ..features.types import UNDEFINED_SEQUENCE_LENGTH, BoolType, MappingType, SequenceType, Type
from ..utils import NestedType
from .base import BaseNode, BaseNodeConfig, RunContext


class CollectNodeConfig(BaseNodeConfig):
    lookup: dict | list | tuple


class CollectNode(BaseNode[CollectNodeConfig]):
    def build_output_type(self, graph: AbstractDataFlowGraph, inputs: dict[str, Reference]) -> Type:
        def _build_type(obj: NestedType[str]):
            if isinstance(obj, dict):
                return MappingType.from_dict({key: _build_type(item) for key, item in obj.items()})

            elif isinstance(obj, (list, tuple)):
                if len(obj) == 0:
                    # default data type for empty sequences
                    dtype = BoolType

                else:
                    dtype, *others = map(_build_type, obj)
                    # check the data types of all sequence items
                    if any(dtype != other for other in others):
                        # TODO: try to cast the values in the sequence to a common type
                        raise NotImplementedError(dtype, others)

                # build the sequence type
                return SequenceType(value_type=dtype, length=len(obj))

            elif isinstance(obj, str):
                # get the data type from the graph
                return graph.get_dtype_from_reference(inputs[obj])

            else:
                raise TypeError(obj)

        return _build_type(self.config.lookup)

    def collect(self, ctx: RunContext, inputs: dict[str, pa.Array]) -> pa.Array:
        def _collect(struct: NestedType[str], dtype: Type) -> pa.Array:
            if isinstance(struct, dict) and isinstance(dtype, MappingType):
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
                assert array_length == dtype.length != UNDEFINED_SEQUENCE_LENGTH
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

            elif isinstance(struct, str):
                # get the referenced input array
                print(struct, type(inputs[struct]))
                return inputs[struct]

            else:
                raise TypeError(struct, dtype)

        return _collect(self.config.lookup, ctx.output_type)

    @property
    def signature(self) -> Any:
        raise EnvironmentError()

    def call(self, *args: Any, **kwargs: Any) -> Any:
        raise EnvironmentError()
