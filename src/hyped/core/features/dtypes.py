"""Data Type System Module.

This module defines a comprehensive type system for representing data types used in a structured
framework. It includes abstractions for primitive types as well as nested structures including
sequence and mapping types.
"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import datasets
import pyarrow as pa
import pydantic
from datasets.features.features import FeatureType, generate_from_arrow_type
from pydantic_core import core_schema


@dataclass(eq=True, frozen=True)
class Type(ABC):
    """Abstract base class for types in the system.

    This class provides the foundation for defining types within the system.
    Subclasses must implement the :code:`arrow_type` property to map their type
    to a corresponding PyArrow data type
    """

    @property
    @abstractmethod
    def arrow_type(self) -> pa.DataType:
        """Abstract property that returns the corresponding PyArrow data type.

        This property must be implemented by subclasses to provide the PyArrow
        data type representation for the specific type.

        Returns:
            pa.DataType: The PyArrow data type representation of the type.
        """
        ...

    @property
    @abstractmethod
    def hf_feature(self) -> datasets.Features:
        """Abstract property that returns the corresponding HuggingFace Feature.

        This property must be implemented by subclasses to provide the HuggingFace
        Feature representation for the specific type.

        Returns:
            datasets.Features: The Hugging Face Feature representation of the type.
        """
        ...

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Generates and returns the pydantic core schema.

        Args:
            cls (type): The class for which to generate the schema.
            source_type (Any): The source type that is being validated.
            handler (pydantic.GetCoreSchemaHandler): A handler function used to generate
                the schema for the class.

        Returns:
            core_schema.CoreSchema: The generated core schema for the :class:`Feature` class,
                indicating that it is an instance schema.
        """
        return core_schema.is_instance_schema(cls)

    @abstractmethod
    def to_dict(self) -> dict:
        """Converts the object to a dictionary representation.

        This method must be implemented by subclasses to provide a consistent
        dictionary serialization format.

        Returns:
            dict: A dictionary representing the object.
        """
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> Type:
        """Constructs an object from its dictionary representation.

        Subclasses must implement this method to enable deserialization
        of the object from a dictionary.

        Args:
            data (dict): The dictionary representation of the object.

        Returns:
            Type: An instance of the type reconstructed from the dictionary.
        """
        ...


@dataclass(eq=True, frozen=True)
class PrimitiveType(Type):
    """Represents a primitive type in the typing system.

    A primitive type corresponds directly to a PyArrow primitive data type such as
    :code:`int32`, :code:`float64`, or :code:`bool`. This class provides methods for
    serialization and deserialization of primitive types.
    """

    _arrow_type: pa.DataType
    """The underlying :code:`PyArrow` data type."""

    @property
    def arrow_type(self) -> pa.DataType:
        """Returns the corresponding :code:`PyArrow` data type for this primitive type.

        Returns:
            pa.DataType: The :code:`PyArrow` data type representation.
        """
        return self._arrow_type

    @property
    def hf_feature(self) -> datasets.Features:
        """Returns the corresponding HuggingFace Feature for this primitive type.

        Returns:
            datasets.Features: The Hugging Face Feature representation of the type.
        """
        return generate_from_arrow_type(self._arrow_type)

    def __str__(self) -> str:
        """Returns the string representation.

        Returns:
            str: A string representation of the :class:`PrimitiveType` instance.
        """
        return str(self._arrow_type).capitalize()  # pragma: not covered

    def to_dict(self) -> dict:
        """Converts the :class:`PrimitiveType` instance to a dictionary representation.

        Returns:
            dict: A dictionary with the following keys:
                - :code:`type` (str): The type identifier ("PrimitiveType").
                - :code:`arrow_type` (str): The string representation of the PyArrow type.
        """
        return {
            "type": "PrimitiveType",
            "arrow_type": str(self.arrow_type),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PrimitiveType":
        """Constructs a PrimitiveType instance from a dictionary representation.

        Args:
            data (dict): A dictionary with the following keys:
                - :code:`type` (str): Must be "PrimitiveType".
                - :code:`arrow_type` (str): The string representation of the PyArrow type.

        Returns:
            PrimitiveType: An instance of the PrimitiveType class.

        Raises:
            ValueError: If the :code:`type` field is not "PrimitiveType".
        """
        if data["type"] != "PrimitiveType":  # pragma: not covered
            raise ValueError("Invalid type for deserialization")
        return ARROW_SCALAR_TYPE_TO_DTYPE_MAPPING[data["arrow_type"]]


# Primitive Types
BoolType = PrimitiveType(pa.bool_())
"""Boolean type."""

StringType = PrimitiveType(pa.utf8())
"""String type."""

# Signed Integer Types
Int8Type = PrimitiveType(pa.int8())
"""8-bit signed integer type."""

Int16Type = PrimitiveType(pa.int16())
"""16-bit signed integer type."""

Int32Type = PrimitiveType(pa.int32())
"""32-bit signed integer type."""

Int64Type = PrimitiveType(pa.int64())
"""64-bit signed integer type."""

# Unsigned Integer Types
UInt8Type = PrimitiveType(pa.uint8())
"""An 8-bit unsigned integer type."""

UInt16Type = PrimitiveType(pa.uint16())
"""16-bit unsigned integer type."""

UInt32Type = PrimitiveType(pa.uint32())
"""32-bit unsigned integer type."""

UInt64Type = PrimitiveType(pa.uint64())
"""64-bit unsigned integer type."""

# Floating Point Types
Float16Type = PrimitiveType(pa.float16())
"""16-bit floating point type."""

Float32Type = PrimitiveType(pa.float32())
"""32-bit floating point type."""

Float64Type = PrimitiveType(pa.float64())
"""64-bit floating point type."""

ARROW_SCALAR_TYPE_TO_DTYPE_MAPPING: dict[str, Type] = {
    str(BoolType.arrow_type): BoolType,
    str(StringType.arrow_type): StringType,
    str(Int8Type.arrow_type): Int8Type,
    str(Int16Type.arrow_type): Int16Type,
    str(Int32Type.arrow_type): Int32Type,
    str(Int64Type.arrow_type): Int64Type,
    str(UInt8Type.arrow_type): UInt8Type,
    str(UInt16Type.arrow_type): UInt16Type,
    str(UInt32Type.arrow_type): UInt32Type,
    str(UInt64Type.arrow_type): UInt64Type,
    str(Float16Type.arrow_type): Float16Type,
    str(Float32Type.arrow_type): Float32Type,
    str(Float64Type.arrow_type): Float64Type,
}


# Constant
UNDEFINED_SEQUENCE_LENGTH = 2**32 - 1
"""A constant representing an undefined sequence length.

This value is used when the length of a sequence is unknown or unspecified.
"""


@dataclass(eq=True, frozen=True)
class ClassLabelType(PrimitiveType):
    """Represents a class label type, typically used for categorical labels in datasets."""

    names: None | tuple[str] = None
    """A tuple of class label names, which must be provided during initialization."""

    # hf datasets maps class label features to int64
    _arrow_type: typing.Final[pa.DataType] = Int64Type.arrow_type
    """The underlying :code:`PyArrow` data type."""

    def __post_init__(self) -> None:
        """Validate the class label type."""
        assert self.names is not None
        assert isinstance(self.names, tuple)

    def __str__(self) -> str:
        """Returns the string representation.

        Returns:
            str: A string representation of the :class:`PrimitiveType` instance.
        """
        return f"ClassLabel(labels={self.names})"  # pragma: not covered

    @property
    def hf_feature(self) -> datasets.Features:
        """Returns the corresponding HuggingFace Feature for this class label type.

        Returns:
            datasets.Features: The Hugging Face Feature representation of the type.
        """
        return datasets.ClassLabel(names=list(self.names))

    def __len__(self) -> int:
        """Returns the number of class labels.

        Returns:
            int: The number of class labels.
        """
        return len(self.names)

    def to_dict(self) -> dict:
        """Converts the :class:`PrimitiveType` instance to a dictionary representation.

        Returns:
            dict: A dictionary with the following keys:
                - :code:`type` (str): The type identifier ("PrimitiveType").
                - :code:`arrow_type` (str): The string representation of the PyArrow type.
        """
        return {"type": "ClassLabelType", "labels": self.names}

    @classmethod
    def from_dict(cls, data: dict) -> "PrimitiveType":
        """Constructs a PrimitiveType instance from a dictionary representation.

        Args:
            data (dict): A dictionary with the following keys:
                - :code:`type` (str): Must be "PrimitiveType".
                - :code:`arrow_type` (str): The string representation of the PyArrow type.

        Returns:
            PrimitiveType: An instance of the PrimitiveType class.

        Raises:
            ValueError: If the :code:`type` field is not "PrimitiveType".
        """
        if data["type"] != "ClassLabelType":  # pragma: not covered
            raise ValueError("Invalid type for deserialization")
        return ClassLabelType(names=tuple(data["labels"]))


@dataclass(eq=True, frozen=True)
class SequenceType(Type, typing.Sequence):
    """Represents a sequence type in the typing system."""

    value_type: Type
    """The type of elements within the sequence."""

    length: int = UNDEFINED_SEQUENCE_LENGTH
    """The length of the sequence

    Set to :code:`UNDEFINED_SEQUENCE_LENGTH` in case of dynamic sequence lengths.
    """

    @property
    def arrow_type(self) -> pa.ListType:
        """Returns the corresponding :code:`PyArrow` data type for this sequence.

        Returns:
            pa.ListType: The :code:`PyArrow` list type representation.
        """
        length = -1 if self.length == UNDEFINED_SEQUENCE_LENGTH else self.length
        return pa.list_(self.value_type.arrow_type, list_size=length)

    @property
    def hf_feature(self) -> datasets.Features:
        """Returns the corresponding HuggingFace Feature for this sequence type.

        Returns:
            datasets.Features: The Hugging Face Feature representation of the type.
        """
        return datasets.Sequence(
            self.value_type.hf_feature,
            length=-1 if self.length == UNDEFINED_SEQUENCE_LENGTH else self.length,
        )

    def _slice_length(self, index: slice) -> int:
        """Calculates the length of the sequence resulting from slicing.

        Args:
            index (slice): The slice to apply to the sequence.

        Returns:
            int: The length of the sliced sequence, or :code:`UNDEFINED_SEQUENCE_LENGTH`
            if the original sequence length is unknown.
        """
        if self.length == UNDEFINED_SEQUENCE_LENGTH:
            return UNDEFINED_SEQUENCE_LENGTH

        # Compute the start, stop, and step of the slice based on the given length
        start, stop, step = index.indices(self.length)

        # If the step is positive, ensure stop is greater than start
        # If the step is negative, ensure stop is less than start
        if (step > 0 and start >= stop) or (step < 0 and start <= stop):
            return 0

        # Compute the length of the sliced sequence
        return max(0, (stop - start + (step - 1 if step > 0 else step + 1)) // step)

    def __len__(self) -> int:
        """Returns the length of the sequence.

        Returns:
            int: The length of the sequence, or :code:`UNDEFINED_SEQUENCE_LENGTH` if unknown.
        """
        return self.length

    @typing.overload
    def __getitem__(self, index: int) -> Type:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> SequenceType:
        ...

    def __getitem__(self, index: int | slice) -> Type:
        """Accesses an element or a subsequence of the sequence.

        Args:
            index (int | slice): The index or slice to access.

        Returns:
            Type | SequenceType: The element type at the given index, or a new
            :class:`SequenceType` representing the subsequence.

        Raises:
            IndexError: If the index is out of bounds for the sequence.
        """
        if isinstance(index, int):
            if index >= self.length:
                raise IndexError(
                    f"Index {index} out of bounds for sequence of length {self.length}"
                )

            return self.value_type

        assert isinstance(index, slice)
        return SequenceType(self.value_type, self._slice_length(index))

    def __eq__(self, other: SequenceType) -> bool:
        """Checks equality between this sequence and another.

        Args:
            other (SequenceType): The other sequence to compare.

        Returns:
            bool: Whether the two sequences are considered equal.
        """
        if not isinstance(other, SequenceType):
            return False

        # TODO: decide if unkown == known length is ok
        return self.value_type == other.value_type and (self.length == other.length)

    def __str__(self) -> str:
        """Returns the string representation.

        Returns:
            str: A string representation of the sequence instance including the value type.
        """
        return f"SequenceType[{str(self.value_type)}]"  # pragma: not covered

    def to_dict(self) -> dict:
        """Serializes the sequence type to a dictionary representation.

        Returns:
            dict: A dictionary containing the serialized information of the sequence type.
        """
        return {
            "type": "SequenceType",
            "value_type": self.value_type.to_dict(),
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SequenceType:
        """Deserializes a dictionary representation into a :class:`SequenceType` instance.

        Args:
            data (dict): A dictionary containing serialized information of a sequence type.

        Returns:
            SequenceType: The deserialized sequence type.

        Raises:
            ValueError: If the dictionary does not represent a SequenceType.
        """
        if data["type"] != "SequenceType":  # pragma: not covered
            raise ValueError("Invalid type for deserialization")
        return SequenceType(
            value_type=build_type_from_dict(data["value_type"]), length=data["length"]
        )


@dataclass(eq=True, frozen=True)
class MappingType(Type, typing.Mapping[str, Type]):
    """Represents a mapping type in the typing system.

    This type associates field names with their corresponding types.
    """

    fields: tuple[tuple[str, Type]]
    """The fields in the mapping, represented as a tuple of key-type pairs."""

    @property
    def arrow_type(self) -> pa.StructType:
        """Returns the corresponding :code:`PyArrow` data type for this mapping.

        Returns:
            pa.StructType: The :code:`PyArrow` struct type representation.
        """
        return pa.struct([(key, field.arrow_type) for key, field in self.fields])

    @property
    def arrow_schema(self) -> pa.Schema:
        """Returns the :code:`PyArrow` schema for this mapping.

        Returns:
            pa.Schema: The :code:`PyArrow` schema representation.
        """
        return pa.schema(self.arrow_type)

    @property
    def hf_feature(self) -> datasets.Features:
        """Returns the corresponding HuggingFace dataset features for this mapping.

        This property converts the PyArrow schema associated with the mapping type into a
        HuggingFace :class:`Features` object.

        Returns:
            datasets.Features: The Hugging Face :class:`Features` object representing
            the mapping type.
        """
        return datasets.Features({key: field.hf_feature for key, field in self.fields})

    def __len__(self) -> int:
        """Returns the number of fields in the mapping.

        Returns:
            int: The number of fields.
        """
        return len(self.fields)

    def __iter__(self) -> typing.Iterable[str]:
        """Iterates over the keys in the mapping.

        Returns:
            Iterable[str]: An iterator over the keys.
        """
        return (key for key, _ in self.fields)

    def __getitem__(self, key: str) -> Type:
        """Gets the type associated with a key in the mapping.

        Args:
            key (str): The key to look up.

        Returns:
            Type: The type associated with the key.

        Raises:
            KeyError: If the key is not present in the mapping.
        """
        return dict(self.fields)[key]

    def __eq__(self, other: MappingType) -> bool:
        """Checks equality between this mapping and another.

        Args:
            other (MappingType): The other mapping to compare.

        Returns:
            bool: Whether the two mappings are considered equal.
        """
        if not isinstance(other, MappingType):
            return False

        # order of fields doesn't matter
        return dict(self.fields) == dict(other.fields)

    @classmethod
    def construct(cls, fields: dict[str, Type]) -> MappingType:
        """Creates a :class:`MappingType` instance from a dictionary of fields.

        Args:
            fields (dict[str, Type]): A dictionary mapping keys to their types.

        Returns:
            MappingType: A new :class:`MappingType` instance.
        """
        return cls(fields=tuple((key, fields[key]) for key in sorted(fields.keys())))

    def __str__(self) -> str:
        """Returns the string representation.

        Returns:
            str: The name of the class (`MappingType`).
        """
        return type(self).__name__  # pragma: not covered

    def to_dict(self) -> dict:
        """Serializes the :class:`MappingType` instance to a dictionary.

        Converts the mapping's fields into a dictionary format,
        suitable for JSON serialization or storage.

        Returns:
            dict: A dictionary representing the mapping type.
        """
        return {
            "type": "MappingType",
            "fields": {key: field.to_dict() for key, field in self.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> MappingType:
        """Deserializes a :class:`MappingType` instance from a dictionary.

        Args:
            data (dict): A dictionary containing the serialized form of a MappingType.

        Returns:
            MappingType: A new instance of :class:`MappingType` constructed from the data.

        Raises:
            ValueError: If the "type" field in the dictionary is not "MappingType".
        """
        if data["type"] != "MappingType":  # pragma: not covered
            raise ValueError("Invalid type for deserialization")
        return MappingType.construct(
            {key: build_type_from_dict(field) for key, field in data["fields"].items()}
        )


def build_type_from_dict(data: dict) -> Type:
    """Constructs a Type instance from a dictionary.

    This function determines the type of the serialized data and calls the appropriate
    `from_dict` method to reconstruct the corresponding `Type` instance.

    Args:
        data (dict): A dictionary representing the serialized form of a Type,
                     containing a "type" field.

    Returns:
        Type: An instance of the appropriate Type subclass.

    Raises:
        ValueError: If the "type" field does not correspond to a recognized Type subclass.
    """
    if data["type"] == "ClassLabelType":
        return ClassLabelType.from_dict(data)
    elif data["type"] == "PrimitiveType":
        return PrimitiveType.from_dict(data)
    elif data["type"] == "SequenceType":
        return SequenceType.from_dict(data)
    elif data["type"] == "MappingType":
        return MappingType.from_dict(data)

    raise ValueError(f"Unknown type for deserialization: {data['type']}")  # pragma: not covered


def build_dtype_from_arrow_type(arrow_type: pa.DataType) -> Type:
    """Build a data type from a given Arrow type.

    Arguments:
        arrow_type (pa.DataType): The Arrow type to convert.

    Returns:
        Type: The corresponding data type.

    Raises:
        TypeError: If the Arrow type is unsupported.
    """
    if pa.types.is_struct(arrow_type):
        fields = map(arrow_type.field, range(arrow_type.num_fields))
        fields = {field.name: build_dtype_from_arrow_type(field.type) for field in fields}
        return MappingType.construct(fields)

    elif pa.types.is_list(arrow_type):
        return SequenceType(
            value_type=build_dtype_from_arrow_type(arrow_type.value_type),
        )

    if (
        isinstance(arrow_type, pa.DataType)
        and str(arrow_type) in ARROW_SCALAR_TYPE_TO_DTYPE_MAPPING
    ):
        return ARROW_SCALAR_TYPE_TO_DTYPE_MAPPING[str(arrow_type)]

    raise TypeError(f"Unsupported type: {arrow_type}")


def build_dtype_from_hf_feature(feature: FeatureType) -> Type:
    """Build a data type from a given Hugging Face feature.

    Arguments:
        feature (FeatureType): The Hugging Face feature to convert.

    Returns:
        Type: The corresponding data type.

    Raises:
        TypeError: If the feature type is unsupported.
    """
    if isinstance(feature, datasets.Features):
        fields = {key: build_dtype_from_hf_feature(field) for key, field in feature.items()}
        return MappingType.construct(fields)

    if isinstance(feature, datasets.Sequence):
        value_type = feature.feature if isinstance(feature, datasets.Sequence) else feature[0]
        length = feature.length if isinstance(feature, datasets.Sequence) else -1
        return SequenceType(
            value_type=build_dtype_from_hf_feature(value_type),
            length=UNDEFINED_SEQUENCE_LENGTH if length == -1 else length,
        )

    elif isinstance(feature, datasets.Value):
        packed = datasets.Features({"field": feature})
        arrow_type = packed.arrow_schema.field("field").type
        return ARROW_SCALAR_TYPE_TO_DTYPE_MAPPING[str(arrow_type)]

    elif isinstance(feature, datasets.ClassLabel):
        return ClassLabelType(names=tuple(feature.names))

    raise TypeError(f"Unsupported feature: {feature}")


PYTHON_PRIMITIVE_TO_DTYPE_MAPPING: dict[type, Type] = {
    bool: BoolType,
    str: StringType,
    int: Int32Type,
    float: Float64Type,
}


def build_dtype_from_python_object(obj: Any) -> Type:
    """Build a data type from a Python object.

    Arguments:
        obj (Any): The object to derive the data type from.

    Returns:
        Type: The corresponding data type.

    Raises:
        TypeError: If the object type is unsupported.
        RuntimeError: If there are inconsistencies in list item types.
    """
    if isinstance(obj, dict):
        return MappingType.construct(
            {key: build_dtype_from_python_object(val) for key, val in obj.items()}
        )

    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            dtype, *others = list(map(build_dtype_from_python_object, obj))
            if any(dtype != other for other in others):
                raise RuntimeError()  # TODO: error message

        else:
            dtype = BoolType

        return SequenceType(value_type=dtype, length=len(obj))

    elif isinstance(obj, tuple(PYTHON_PRIMITIVE_TO_DTYPE_MAPPING.keys())):
        return PYTHON_PRIMITIVE_TO_DTYPE_MAPPING[type(obj)]

    raise TypeError(f"Unsupported object: {obj}")


def is_dtype_subset(dtype_a: Type, dtype_b: Type) -> bool:
    """Recursively checks if dtype_a is a subset of dtype_b.

    Arguments:
        dtype_a (Type): The type that should be a subset.
        dtype_b (Type): The type that should be a superset.

    Returns:
        bool: True if dtype_a is a subset of dtype_b, False otherwise.
    """
    if isinstance(dtype_a, MappingType) and isinstance(dtype_b, MappingType):
        return all(
            (key in dtype_b) and is_dtype_subset(dtype, dtype_b[key])
            for key, dtype in dtype_a.items()
        )

    elif isinstance(dtype_a, SequenceType) and isinstance(dtype_b, SequenceType):
        return (dtype_a.length == dtype_b.length) and is_dtype_subset(
            dtype_a.arrow_type, dtype_b.arrow_type
        )

    return dtype_a == dtype_b


def cast_dtype(src_dtype: Type, tgt_dtype: Type) -> Type:
    """Perform type casting of a source to a target data type.

    This function attempts to cast a source data type (:code:`src_dtype`) to a target data type
    (:code:`tgt_dtype`). It supports casting between various data types such as sequences,
    mappings, and primitive types. If the casting is not feasible due to type mismatches
    or constraints (e.g., incompatible lengths for sequences or mismatched keys for
    mappings), an exception is raised.

    Args:
        src_dtype (Type): The source data type.
        tgt_dtype (Type): The target data type.

    Returns:
        Type: The resulting data type after a successful cast.

    Raises:
        RuntimeError: If the casting operation is not feasible due to type
            mismatches or constraints.
    """
    if isinstance(src_dtype, SequenceType) and isinstance(tgt_dtype, SequenceType):
        if (
            len(src_dtype) != UNDEFINED_SEQUENCE_LENGTH
            and len(tgt_dtype) != UNDEFINED_SEQUENCE_LENGTH
            and len(src_dtype) != len(tgt_dtype)
        ):
            raise RuntimeError(
                f"Cannot cast sequence of length {len(src_dtype)} to a "
                f"sequence of length {len(tgt_dtype)}."
            )

        length = len(src_dtype) if len(src_dtype) != UNDEFINED_SEQUENCE_LENGTH else len(tgt_dtype)

        return SequenceType(cast_dtype(src_dtype.value_type, tgt_dtype.value_type), length=length)

    elif isinstance(src_dtype, MappingType) and isinstance(tgt_dtype, MappingType):
        if set(src_dtype.keys()) != set(tgt_dtype.keys()):
            raise RuntimeError(
                f"Cannot cast mapping with keys {set(src_dtype.keys())} to a "
                f"mapping with keys {set(tgt_dtype.keys())}."
            )

        # make sure both mappings contain all keys and the fields
        # are castable
        fields = {k: cast_dtype(src_dtype[k], tgt_dtype[k]) for k in src_dtype.keys()}
        return MappingType.construct(fields)

    elif isinstance(src_dtype, PrimitiveType) and isinstance(tgt_dtype, PrimitiveType):
        if not isinstance(src_dtype, ClassLabelType) and isinstance(tgt_dtype, ClassLabelType):
            # cannot cast non-class-label type to class-label type
            raise RuntimeError(
                f"Cannot cast type {src_dtype} to ClassLabelType. "
                "Only other ClassLabelTypes can be cast to ClassLabelType."
            )

        elif (
            isinstance(src_dtype, ClassLabelType) and isinstance(tgt_dtype, ClassLabelType)
        ) and len(src_dtype) != len(tgt_dtype):
            # TODO: should we allow target labels to be a subset of the source labels
            raise RuntimeError(
                f"Cannot cast ClassLabelType with {len(src_dtype)} labels to "
                f"ClassLabelType with {len(tgt_dtype)} labels. Both label sets must have the "
                f"same number of labels."
            )

        # can cast all primitive types to all other primitive type
        return tgt_dtype

    raise RuntimeError(
        f"Cannot cast type {type(src_dtype).__name__} to {type(tgt_dtype).__name__}."
    )


def common_dtype(*dtypes: Type) -> Type:
    """Determine the common dtype for a set of input dtypes.

    This function identifies a dtype to which all input dtypes can be cast
    without loss of information, based on a predefined priority hierarchy.
    It supports primitive types, sequences, and mappings.

    Args:
        dtypes (Type): The input dtypes to compare. These can be :class:`PrimitiveType`,
            :class:`SequenceType`, or :class:`MappingType` objects.

    Returns:
        Type: The common dtype that can represent all input dtypes.

    Raises:
        AssertionError: If no dtypes are provided or the dtypes cannot be combined.
        RuntimeError: If dtypes are incompatible or cannot be resolved to a common dtype.
    """
    # Ensure at least one dtype is provided
    assert len(dtypes) > 0, "At least one dtype must be provided to determine a common dtype."

    if len(dtypes) == 1:
        return dtypes[0]

    # convert all class label types to integer types first
    dtypes = tuple(Int64Type if isinstance(dtype, ClassLabelType) else dtype for dtype in dtypes)

    PRIORITY = {  # noqa: N806
        BoolType: 0,
        Int8Type: 1,
        UInt8Type: 2,
        Int16Type: 3,
        UInt16Type: 4,
        Int32Type: 5,
        UInt32Type: 6,
        Int64Type: 7,
        UInt64Type: 8,
        Float16Type: 9,
        Float32Type: 10,
        Float64Type: 11,
        StringType: 12,
    }

    # Handle sequence types
    if all(isinstance(dtype, SequenceType) for dtype in dtypes):
        # Check for common sequence length
        unique_lengths = set(map(len, dtypes))
        length = (
            UNDEFINED_SEQUENCE_LENGTH if len(unique_lengths) > 1 else next(iter(unique_lengths))
        )
        # Determine common value type
        value_type = common_dtype(*(dtype.value_type for dtype in dtypes))
        return SequenceType(value_type, length)

    # Handle mapping types
    elif all(isinstance(dtype, MappingType) for dtype in dtypes):
        # Ensure all mappings have the same keys
        keys = set(dtypes[0].keys())
        if any(keys != set(dtype.keys()) for dtype in dtypes[1:]):
            raise RuntimeError(
                "MappingType dtypes must have identical keys to determine a common dtype."
            )
        # Determine common dtype for each field
        fields = {k: common_dtype(*(dtype[k] for dtype in dtypes)) for k in keys}
        return MappingType.construct(fields)

    # Handle primitive types
    elif all(isinstance(dtype, PrimitiveType) for dtype in dtypes):
        # get the data type with the highest priority
        assert all(dtype in PRIORITY for dtype in dtypes), "Unsupported dtype encountered"
        dtype = max(dtypes, key=PRIORITY.__getitem__)

        int_types = {Int8Type, Int16Type, Int32Type, Int64Type}
        # promote data type if required
        if (dtype == UInt8Type) and any(typ in dtypes for typ in int_types):
            return Int16Type
        elif (dtype == UInt16Type) and any(typ in dtypes for typ in int_types):
            return Int32Type
        elif (dtype == UInt32Type) and any(typ in dtypes for typ in int_types):
            return Int64Type
        elif (dtype == UInt64Type) and any(typ in dtypes for typ in int_types):
            return Int64Type
        else:
            return dtype

    # Raise error for incompatible dtypes
    raise RuntimeError("Dtypes are incompatible or cannot be resolved to a common dtype. ")
