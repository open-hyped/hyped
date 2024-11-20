"""Core typing module.

This module defines the type aliases that need to me used to define
node interfaces.
"""
from typing import Annotated, Any, TypeAlias, TypeVar, Union

import pyarrow as pa

from hyped.common._worker import Rank

from .features.features import BoolFeature
from .features.features import Feature as _Feature
from .features.features import (
    Float16Feature,
    Float32Feature,
    Float64Feature,
    Int8Feature,
    Int16Feature,
    Int32Feature,
    Int64Feature,
    MappingFeature,
    SequenceFeature,
    StringFeature,
    UInt8Feature,
    UInt16Feature,
    UInt32Feature,
    UInt64Feature,
)
from .features.reference import NodeId
from .features.validators import Len

__all__ = [
    "Index",
    "IndexList",
    "TraceIndexList",
    "NodeId",
    "PartitionId",
    "Rank",
    "Union",
    "Annotated",
    "Feature",
    "String",
    "Bool",
    "Int",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float",
    "Float16",
    "Float32",
    "Float64",
    "Sequence",
    "Mapping",
    "Len",
]

Index: TypeAlias = int
"""An index, usually corresponding to a sample.

Represents a single integer that refers to a specific sample within the dataset. This is often used
to retrieve or reference a particular sample from a dataset.
"""

IndexList: TypeAlias = list[Index]
"""A list of dataset indices, usually corresponding to a batch.

Contains integer indices that refer to specific samples within the dataset.
This is typically used to track which samples are included in a particular
batch or subset of the dataset.
"""

TraceIndexList: TypeAlias = list[int]
"""A list of trace indices used to map outputs to their source samples in augmentation processes.

In data augmentation, a single input sample can generate multiple output samples. The
:class:`TraceIndexList` tracks the origin of each output sample by maintaining a list of indices.
Each index in this list corresponds to the position of the input sample in the original batch that
was used to generate the output sample.

For example, if :code:`trace_index[i] = j`, it indicates that the `i`-th output sample was derived
from the  :code:`j`-th input sample in the original batch.

Usage Context:
    - When a batch of input samples undergoes augmentation, this list provides a direct mapping
      from each output sample back to its corresponding input sample.
    - This type is commonly returned alongside the augmented batch, enabling users to track which
      input sample produced which output sample.
"""

PartitionId: TypeAlias = str
"""An identifier for a partition within the data flow graph.

The :class:`PartitionId` is a string that uniquely identifies these partitions, enabling the
tracking and management of different stages within the data flow graph.

A partition in the data flow graph represents a subgraph where each sample from the dataset is
processed or transformed independently of others. Partitions are often introduced during data
augmentation processes, where new samples are generated or existing samples are filtered out.
"""


Feature: TypeAlias = Union[_Feature, Any, list[Any], pa.Scalar, pa.Array]
"""
Feature: Type alias for a feature.

Supported types include:
- Feature Types: :class:`Feature`
- Built-in types: :class:`Any`, :class:`list[Any]`
- PyArrow scalar types: :class:`pyarrow.Scalar`
- PyArrow array types: :class:`pyarrow.Array`
"""

String: TypeAlias = Union[StringFeature, str, list[str], pa.StringScalar, pa.StringArray]
"""
String: Type alias for a string.

Supported types include:
- Feature Types: :class:`StringFeature`
- Built-in types: :class:`str`, :class:`list[str]`
- PyArrow scalar types: :class:`pyarrow.StringScalar`
- PyArrow array types: :class:`pyarrow.StringArray`
"""

Bool: TypeAlias = Union[BoolFeature, bool, list[bool], pa.BooleanScalar, pa.BooleanArray]
"""
Bool: Type alias for a boolean.

Supported types include:
- Feature Types: :class:`BoolFeature`
- Built-in types: :class:`bool`, :class:`list[bool]`
- PyArrow scalar types: :class:`pyarrow.BooleanScalar`
- PyArrow array types: :class:`pyarrow.BooleanArray`
"""

UInt8: TypeAlias = Union[UInt8Feature, int, list[int], pa.UInt8Scalar, pa.UInt8Array]
"""
UInt8: Type alias for an 8-bit unsigned integer.

Supported types include:
- Feature Types: :class:`UInt8Feature`
- Built-in types: :class:`int`, :class:`list[int]`
- PyArrow scalar types: :class:`pyarrow.UInt8Scalar`
- PyArrow array types: :class:`pyarrow.UInt8Array`
"""

UInt16: TypeAlias = Union[UInt16Feature, int, list[int], pa.UInt16Scalar, pa.UInt16Array]
"""
UInt16: Type alias for a 16-bit unsigned integer.

Supported types include:
- Feature Types: :class:`UInt16Feature`
- Built-in types: :class:`int`, :class:`list[int]`
- PyArrow scalar types: :class:`pyarrow.UInt16Scalar`
- PyArrow array types: :class:`pyarrow.UInt16Array`
"""

UInt32: TypeAlias = Union[UInt32Feature, int, list[int], pa.UInt32Scalar, pa.UInt32Array]
"""
UInt32: Type alias for a 32-bit unsigned integer.

Supported types include:
- Feature Types: :class:`UInt32Feature`
- Built-in types: :class:`int`, :class:`list[int]`
- PyArrow scalar types: :class:`pyarrow.UInt32Scalar`
- PyArrow array types: :class:`pyarrow.UInt32Array`
"""

UInt64: TypeAlias = Union[UInt64Feature, int, list[int], pa.UInt64Scalar, pa.UInt64Array]
"""
UInt64: Type alias for a 64-bit unsigned integer.

Supported types include:
- Feature Types: :class:`UInt64Feature`
- Built-in types: :class:`int`, :class:`list[int]`
- PyArrow scalar types: :class:`pyarrow.UInt64Scalar`
- PyArrow array types: :class:`pyarrow.UInt64Array`
"""

UInt: TypeAlias = Union[
    UInt8Feature,
    UInt16Feature,
    UInt32Feature,
    UInt64Feature,
    int,
    list[int],
    pa.UInt8Scalar,
    pa.UInt16Scalar,
    pa.UInt32Scalar,
    pa.UInt64Scalar,
    pa.UInt8Array,
    pa.UInt16Array,
    pa.UInt32Array,
    pa.UInt64Array,
]
"""
UInt: Type alias for an unsigned integer of varying bit length.

Supported types include:
- Feature Types: :class:`UInt8Feature`, :class:`UInt16Feature`, :class:`UInt32Feature`,
  :class:`UInt64Feature`
- Built-in types: :class:`int`, :class:`list[int]`
- PyArrow scalar types: :class:`pyarrow.UInt8Scalar`, :class:`pyarrow.UInt16Scalar`,
  :class:`pyarrow.UInt32Scalar`, :class:`pyarrow.UInt64Scalar`
- PyArrow array types: :class:`pyarrow.UInt8Array`, :class:`pyarrow.UInt16Array`,
  :class:`pyarrow.UInt32Array`, :class:`pyarrow.UInt64Array`
"""

Int8: TypeAlias = Union[Int8Feature, int, list[int], pa.Int8Scalar, pa.Int8Array]
"""
Int8: Type alias for an 8-bit signed integer.

Supported types include:
- Feature Types: :class:`Int8Feature`
- Built-in types: :class:`int`, :class:`list[int]`
- PyArrow scalar types: :class:`pyarrow.Int8Scalar`
- PyArrow array types: :class:`pyarrow.Int8Array`
"""

Int16: TypeAlias = Union[Int16Feature, int, list[int], pa.Int16Scalar, pa.Int16Array]
"""
Int16: Type alias for a 16-bit signed integer.

Supported types include:
- Feature Types: :class:`Int16Feature`
- Built-in types: :class:`int`, :code:`list[int]`
- PyArrow scalar types: :class:`pyarrow.Int16Scalar`
- PyArrow array types: :class:`pyarrow.Int16Array`
"""

Int32: TypeAlias = Union[Int32Feature, int, list[int], pa.Int32Scalar, pa.Int32Array]
"""
Int32: Type alias for a 32-bit signed integer.

Supported types include:
- Feature Types: :class:`Int32Feature`
- Built-in types: :class:`int`, :code:`list[int]`
- PyArrow scalar types: :class:`pyarrow.Int32Scalar`
- PyArrow array types: :class:`pyarrow.Int32Array`
"""

Int64: TypeAlias = Union[Int64Feature, int, list[int], pa.Int64Scalar, pa.Int64Array]
"""
Int64: Type alias for a 64-bit signed integer.

Supported types include:
- Feature Types: :class:`Int64Feature`
- Built-in types: :class:`int`, :code:`list[int]`
- PyArrow scalar types: :class:`pyarrow.Int64Scalar`
- PyArrow array types: :class:`pyarrow.Int64Array`
"""

Int: TypeAlias = Union[
    Int8Feature,
    Int16Feature,
    Int32Feature,
    Int64Feature,
    int,
    list[int],
    pa.Int8Scalar,
    pa.Int16Scalar,
    pa.Int32Scalar,
    pa.Int64Scalar,
    pa.Int8Array,
    pa.Int16Array,
    pa.Int32Array,
    pa.Int64Array,
]
"""
Int: Type alias for a signed integer of varying bit length.

Supported types include:
- Feature Types: :class:`Int8Feature`, :class:`Int16Feature`, :class:`Int32Feature`,
  :class:`Int64Feature`
- Built-in types: :class:`int`, :code:`list[int]`
- PyArrow scalar types: :class:`pyarrow.Int8Scalar`, :class:`pyarrow.Int16Scalar`,
  :class:`pyarrow.Int32Scalar`, :class:`pyarrow.Int64Scalar`
- PyArrow array types: :class:`pyarrow.Int8Array`, :class:`pyarrow.Int16Array`,
  :class:`pyarrow.Int32Array`, :class:`pyarrow.Int64Array`
"""

Float16: TypeAlias = Union[
    Float16Feature, float, list[float], pa.HalfFloatScalar, pa.HalfFloatArray
]
"""
Float16: Type alias for a 16-bit floating-point number.

Supported types include:
- Feature Types: :class:`Float16Feature`
- Built-in types: :class:`float`, :code:`list[float]`
- PyArrow scalar types: :class:`pyarrow.HalfFloatScalar`
- PyArrow array types: :class:`pyarrow.HalfFloatArray`
"""

Float32: TypeAlias = Union[Float32Feature, float, list[float], pa.FloatScalar, pa.FloatArray]
"""
Float32: Type alias for a 32-bit floating-point number.

Supported types include:
- Feature Types: :class:`Float32Feature`
- Built-in types: :class:`float`, :code:`list[float]`
- PyArrow scalar types: :class:`pyarrow.FloatScalar`
- PyArrow array types: :class:`pyarrow.FloatArray`
"""

Float64: TypeAlias = Union[Float64Feature, float, list[float], pa.DoubleScalar, pa.DoubleArray]
"""
Float64: Type alias for a 64-bit floating-point number.

Supported types include:
- Feature Types: :class:`Float64Feature`
- Built-in types: :class:`float`, :code:`list[float]`
- PyArrow scalar types: :class:`pyarrow.DoubleScalar`
- PyArrow array types: :class:`pyarrow.DoubleArray`
"""

Float: TypeAlias = Union[
    Float16Feature,
    Float32Feature,
    Float64Feature,
    float,
    list[float],
    pa.HalfFloatScalar,
    pa.FloatScalar,
    pa.DoubleScalar,
    pa.HalfFloatArray,
    pa.FloatArray,
    pa.DoubleArray,
]
"""
Float: Type alias for a floating-point number of varying precision.

Supported types include:
- Feature Types: :class:`Float16Feature`, :class:`Float32Feature`, :class:`Float64Feature`
- Built-in types: :class:`float`, :code:`list[float]`
- PyArrow scalar types: :class:`pyarrow.HalfFloatScalar`, :class:`pyarrow.FloatScalar`,
  :class:`pyarrow.DoubleScalar`
- PyArrow array types: :class:`pyarrow.HalfFloatArray`, :class:`pyarrow.FloatArray`,
  :class:`pyarrow.DoubleArray`
"""

T = TypeVar("T")

Sequence: TypeAlias = Union[SequenceFeature[T], list[T], list[list[T]], pa.ListScalar, pa.ListArray]
"""
Sequence: Type alias for a sequence of items of a specified type.

Supported types include:
- Feature Types: :code:`SequenceFeature[T]`
- Built-in types: :code:`list[T]`, :code:`list[list[T]]`
- PyArrow scalar types: :class:`pyarrow.ListScalar`
- PyArrow array types: :class:`pyarrow.ListArray`
"""

Mapping: TypeAlias = MappingFeature
