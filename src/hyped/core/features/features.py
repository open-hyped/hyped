"""Features Module.

This module defines the core classes and utilities for managing and validating different types of
features. Features represent various data types and their structure, validated and resolved using
Pydantic schemas. This allows users to define flexible and complex data structures while ensuring
type safety and compatibility with the data flow system.

Features act as the high-level interface for describing data in a structured and extensible manner.
Each feature is associated with a reference (:class:`BaseReference`) that links it to a specific
node or placeholder in the graph, providing precise context and origin information. These
references can point to concrete nodes or act as forward declarations for features yet to be
defined.

A core functionality of features is their ability to resolve complex type annotations, such as
unions or nested structures, into concrete feature instances. Pydantic handles this resolution
during validation, ensuring that users can define dynamic or multi-type features with confidence.
This capability is particularly valuable when dealing with nested schemas or type unions, as it
ensures the correct feature type is inferred and instantiated.

Additionally, the module includes utilities for registering and executing feature-specific methods
dynamically, allowing for extensible functionality without introducing tight coupling or circular
dependencies.
"""

from __future__ import annotations

import typing
import warnings
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from types import GenericAlias
from typing import Any, Callable, ClassVar, Final, Generic, TypeVar, overload

import pydantic
from pydantic.type_adapter import _type_has_config
from pydantic_core import PydanticCustomError, core_schema
from typing_extensions import Self

from hyped.common._pydantic import BaseModelWithArbitraryTypesAllowed
from hyped.common.utils import is_python_version_less_than

from . import dtypes
from .mixins import MethodRegistryMixin
from .reference import BaseReference, ForwardReference

if is_python_version_less_than(3, 12):  # pragma: not covered

    def get_original_bases(cls: type) -> type:
        """Return the class's "original" bases prior to modification by :code:`__mro_entries__`."""
        try:
            return cls.__dict__.get("__orig_bases__", cls.__bases__)
        except AttributeError:
            raise TypeError(f"Expected an instance of type, got {type(cls).__name__!r}") from None

else:
    from types import get_original_bases  # noqa: E402


DataType = TypeVar("DataType", bound=dtypes.Type)


@dataclass(eq=True, frozen=False)
class Feature(MethodRegistryMixin, Generic[DataType]):
    """Base class for defining features in a data flow graph.

    A feature serves as the primary user interface for defining and working with structured data
    in a data flow system. It wraps a reference instance (:class:`BaseReference`) that describes
    its origin in the graph, enabling precise modeling of the flow of data between nodes. The
    data type (`dtype`) is inferred directly from the reference.

    Features are responsible for:

    1. **Validation:** Features use Pydantic schemas to validate the data structure and type,
       ensuring compatibility with the defined feature.

    2. **Resolution:** Features can be resolved to concrete instances using Pydantic's
       validation mechanism. This resolution process is critical when users define
       complex data structures, such as unions of features or nested structures.
       Pydantic ensures that these annotations are validated and resolved to the
       appropriate concrete feature type.

    3. **Extensibility:** Through the dynamic method registry, features support the
       addition of custom methods for extended functionality. This decouples
       feature definitions from specific implementation details, avoiding circular dependencies.
    """

    ref: BaseReference
    """The reference to the feature."""

    @property
    def dtype(self) -> DataType:
        """The data type of the feature inferred from the reference."""
        dtype = self.ref.get_dtype()
        assert dtype is not None
        return dtype

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


@dataclass(eq=True, frozen=False)
class PrimitiveFeature(Feature[DataType]):
    """Base class for primitive feature types.

    The :class:`PrimitiveFeature` class extends :class:`Feature` to represent features
    with primitive data types.
    """

    _expected_dtype: ClassVar[dtypes.Type]
    """The expected data type for this feature.

    Subclasses should define this attribute to indicate the primitive data type
    that the feature represents.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Builds the Pydantic core schema for primitive feature types.

        Validates and enforces the expected data type for a feature. If the input
        is a :class:`BaseReference`, it is converted to the corresponding feature type.

        Args:
            source_type (Any): The type being validated.
            handler (pydantic.GetCoreSchemaHandler): A handler function used to the core schema.

        Returns:
            core_schema.CoreSchema: The Pydantic core schema for the mapping feature,
                which is used to validate instances of the mapping.

        Raises:
            PydanticCustomError: If the data type of the instance does not match
                the expected :code:`dtype`.
        """

        def validator_fn(inst: Feature | BaseReference, validator: Callable[[Any], Any]) -> Feature:
            if isinstance(inst, PrimitiveFeature):
                # convert the generic primitive instance to a
                # specific feature type matching the provided data type
                inst = build_feature_from_reference(inst.ref)

            elif isinstance(inst, BaseReference):
                if inst.get_dtype() is None:
                    # reference doesn't specify a data type
                    assert isinstance(inst, ForwardReference)
                    inst = ForwardReference(cls._expected_dtype)

                # create primitve feature with expected data type from reference instance
                inst = build_feature_from_reference(inst)

            # run core validator
            inst = validator(inst)

            # make sure the data type matches the expectation
            if inst.dtype is not cls._expected_dtype:
                # convert class label feature to underlying integer feature
                if isinstance(inst, ClassLabelFeature) and (
                    cls._expected_dtype is dtypes.Int64Type
                ):
                    return Int64Feature(inst.ref)

                raise PydanticCustomError(
                    "Type Mismatch",
                    "Data type '{actual}' doesn't match expected data type '{expected}'",
                    {"actual": inst.dtype, "expected": cls._expected_dtype},
                )

            return inst

        return core_schema.no_info_wrap_validator_function(
            validator_fn, schema=core_schema.is_instance_schema(cls)
        )


class BoolFeature(PrimitiveFeature[dtypes.BoolType]):
    """A primitive feature representing a boolean value."""

    _expected_dtype: Final[dtypes.Type] = dtypes.BoolType

    def __invert__(self) -> BoolFeature:
        """Performs boolean negation ('~').

        Returns:
            BoolFeature: A new :class:`BoolFeature` with the negated value.
        """
        return self.execute_method("__invert__")

    def __and__(self, other: Any) -> BoolFeature:
        """Performs boolean 'and' operation ('&').

        Args:
            other (Any): The boolean value to perform the 'and' operation with.

        Returns:
            BoolFeature: A new :class:`BoolFeature` resulting from the 'and' operation.
        """
        return self.execute_method("__and__", other)

    def __rand__(self, other: Any) -> BoolFeature:
        """Performs right boolean 'and' operation ('&').

        Args:
            other (Any): The boolean value to perform the 'and' operation with.

        Returns:
            BoolFeature: A new :class:`BoolFeature` resulting from the right 'and' operation.
        """
        return self.execute_method("__rand__", other)

    def __or__(self, other: Any) -> BoolFeature:
        """Performs boolean 'or' operation ('|').

        Args:
            other (Any): The boolean value to perform the 'or' operation with.

        Returns:
            BoolFeature: A new :class:`BoolFeature` resulting from the 'or' operation.
        """
        return self.execute_method("__or__", other)

    def __ror__(self, other: Any) -> BoolFeature:
        """Performs right boolean 'or' operation ('|').

        Args:
            other (Any): The boolean value to perform the 'or' operation with.

        Returns:
            BoolFeature: A new :class:`BoolFeature` resulting from the right 'or' operation.
        """
        return self.execute_method("__ror__", other)

    def __xor__(self, other: Any) -> BoolFeature:
        """Performs boolean 'exclusive or' operation ('^').

        Args:
            other (Any): The boolean value to perform the 'exclusive or' operation with.

        Returns:
            BoolFeature: A new :class:`BoolFeature` resulting from the 'exclusive or' operation.
        """
        return self.execute_method("__xor__", other)

    def __rxor__(self, other: Any) -> BoolFeature:
        """Performs right boolean 'exclusive or' operation ('^').

        Args:
            other (Any): The boolean value to perform the 'exclusive or' operation with.

        Returns:
            BoolFeature: A new :class:`BoolFeature` resulting from the right 'exclusive or'
            operation.
        """
        return self.execute_method("__rxor__", other)


class StringFeature(PrimitiveFeature[dtypes.StringType]):
    """A primitive feature representing a string value."""

    _expected_dtype: Final[dtypes.Type] = dtypes.StringType

    def __add__(self, other: Any) -> StringFeature:
        """Performs string concatenation ('+').

        Args:
            other (Any): The value to concatenate to this string.

        Returns:
            StringFeature: A new :class:`StringFeature` resulting from the concatenation.
        """
        return self.execute_method("__add__", other)

    def __radd__(self, other: Any) -> StringFeature:
        """Performs right string concatenation ('+').

        Args:
            other (Any): The value to concatenate to this string.

        Returns:
            StringFeature: A new :class:`StringFeature` resulting from the concatenation.
        """
        return self.execute_method("__radd__", other)

    def __mul__(self, other: Any) -> StringFeature:
        """Performs string repetition ('*').

        Args:
            other (Any): The number of repetitions.

        Returns:
            StringFeature: A new :class:`StringFeature` with the repeated string.
        """
        return self.execute_method("__mul__", other)

    def __rmul__(self, other: Any) -> StringFeature:
        """Performs right string repetition ('*').

        Args:
            other (Any): The number of repetitions.

        Returns:
            StringFeature: A new :class:`StringFeature` with the repeated string.
        """
        return self.execute_method("__rmul__", other)

    def __getitem__(self, idx: int | slice) -> StringFeature:
        """Performs string slicing or indexing ('[]').

        Args:
            idx (int | slice): The index or slice to extract from the string.

        Returns:
            StringFeature: A new :class:`StringFeature` representing the sliced or indexed value.
        """
        return self.execute_method("__getitem__", idx)

    def __setitem__(self, idx: int | slice, replacement: Any) -> None:
        """Sets a substring or slice ('[] =').

        Args:
            idx (int | slice): The index or slice to replace in the string.
            replacement (Any): The replacement value.

        Returns:
            None
        """
        self.ref = self.execute_method("__setitem__", idx, replacement).ref

    def upper(self) -> StringFeature:
        """Converts the string to uppercase.

        Returns:
            StringFeature: A new :class:`StringFeature` with all characters in uppercase.
        """
        return self.execute_method("upper")

    def lower(self) -> StringFeature:
        """Converts the string to lowercase.

        Returns:
            StringFeature: A new :class:`StringFeature` with all characters in lowercase.
        """
        return self.execute_method("lower")

    def capitalize(self) -> StringFeature:
        """Capitalizes the string (first character uppercase, others lowercase).

        Returns:
            StringFeature: A new :class:`StringFeature` with the string capitalized.
        """
        return self.execute_method("capitalize")

    def title(self) -> StringFeature:
        """Converts the string to title case.

        Returns:
            StringFeature: A new :class:`StringFeature` with each word capitalized.
        """
        return self.execute_method("title")

    def swapcase(self) -> StringFeature:
        """Swaps the case of all characters in the string.

        Returns:
            StringFeature: A new :class:`StringFeature` with case-swapped characters.
        """
        return self.execute_method("swapcase")

    def startswith(self, pattern: str) -> BoolFeature:
        """Checks if the string starts with the specified pattern.

        Args:
            pattern (str): The prefix to check for.

        Returns:
            BoolFeature: True if the string starts with the pattern, otherwise False.
        """
        return self.execute_method("startswith", pattern)

    def endswith(self, pattern: str) -> BoolFeature:
        """Checks if the string ends with the specified pattern.

        Args:
            pattern (str): The suffix to check for.

        Returns:
            BoolFeature: True if the string ends with the pattern, otherwise False.
        """
        return self.execute_method("endswith", pattern)

    def replace(self, pattern: str, replacement: str) -> StringFeature:
        """Replaces occurrences of a pattern with a replacement string.

        Args:
            pattern (str): The substring to replace.
            replacement (str): The replacement string.

        Returns:
            StringFeature: A new :class:`StringFeature` with replacements applied.
        """
        return self.execute_method("replace", pattern, replacement)

    def find(self, pattern: str) -> Int32Feature:
        """Finds the first occurrence of a pattern in the string.

        Note that the resulting index presents the index of the first occurance of the
        pattern in the string **in bytes**. It give unexpected results depending on the
        string encoding.

        Args:
            pattern (str): The substring to search for.

        Returns:
            Int32Feature: The index of the first occurrence of the pattern, in bytes, or -1 if
            not found.
        """
        return self.execute_method("find", pattern)

    def split(
        self, pattern: str = " ", maxsplits: None | int = None
    ) -> SequenceFeature[StringFeature]:
        """Splits the string by a delimiter.

        Args:
            pattern (str): The delimiter to split on. Defaults to a space.
            maxsplits (None | int): The maximum number of splits to perform. Defaults to None.

        Returns:
            SequenceFeature[StringFeature]: A sequence of substrings resulting from the split.
        """
        return self.execute_method("split", pattern, maxsplits, False)

    def rsplit(
        self, pattern: str = " ", maxsplits: None | int = None
    ) -> SequenceFeature[StringFeature]:
        """Splits the string by a delimiter from the right.

        Args:
            pattern (str): The delimiter to split on. Defaults to a space.
            maxsplits (None | int): The maximum number of splits to perform. Defaults to None.

        Returns:
            SequenceFeature[StringFeature]: A sequence of substrings resulting from the split.
        """
        return self.execute_method("split", pattern, maxsplits, True)

    def strip(self, characters: str = " ") -> StringFeature:
        """Strips leading and trailing characters from the string.

        Args:
            characters (str): The characters to strip. Defaults to whitespace.

        Returns:
            StringFeature: A new :class:`StringFeature` with stripped characters.
        """
        return self.execute_method("strip", characters)

    def lstrip(self, characters: str = " ") -> StringFeature:
        """Strips leading characters from the string.

        Args:
            characters (str): The characters to strip. Defaults to whitespace.

        Returns:
            StringFeature: A new :class:`StringFeature` with leading characters stripped.
        """
        return self.execute_method("lstrip", characters)

    def rstrip(self, characters: str = " ") -> StringFeature:
        """Strips trailing characters from the string.

        Args:
            characters (str): The characters to strip. Defaults to whitespace.

        Returns:
            StringFeature: A new :class:`StringFeature` with trailing characters stripped.
        """
        return self.execute_method("rstrip", characters)

    def format(self, *args: Any, **kwargs: Any) -> StringFeature:
        """Formats the string using provided positional and keyword arguments.

        Args:
            *args (Any): Positional arguments for formatting.
            **kwargs (Any): Keyword arguments for formatting.

        Returns:
            StringFeature: The formatted string feature.
        """
        return self.execute_method("format", *args, **kwargs)


class StatisticalFeatureMixin:
    """Mixin class that adds statistical operations for features.

    This mixin provides common statistical operations that can be applied
    to primitive features, such as :code:`sum` and :code:`mean`. These
    operations calculate aggregate values for the feature, helping to
    analyze and summarize the data.
    """

    def sum(self: PrimitiveFeature) -> Self:
        """Calculates the sum of the feature's values.

        This method computes the total sum of the values associated with
        the feature. It is intended for use with numerical data types.

        Returns:
            Self: A new feature representing the sum of the original feature's values.
        """
        return self.execute_method("sum")

    def mean(self: PrimitiveFeature) -> Float64Feature:
        """Calculates the mean (average) of the feature's values.

        This method computes the average of the values associated with the
        feature. It is intended for use with numerical data types.

        Returns:
            Float64Feature: A new feature representing the mean of the original feature's values.
        """
        return self.execute_method("mean")


class ComparableFeatureMixin:
    """Mixin providing comparison operations for primitive features.

    This mixin defines methods for comparing primitive features and for obtaining
    minimum and maximum values. Subclasses implementing primitive features can
    extend their functionality by inheriting from this mixin.
    """

    def min(self: PrimitiveFeature) -> Self:
        """Returns the minimum value of the feature.

        Returns:
            Self: A new feature instance representing the minimum value.
        """
        return self.execute_method("min")

    def max(self: PrimitiveFeature) -> Self:
        """Returns the maximum value of the feature.

        Returns:
            Self: A new feature instance representing the maximum value.
        """
        return self.execute_method("max")

    def __eq__(self: PrimitiveFeature, other: Any) -> BoolFeature:
        """Compares if the feature is equal to another value.

        Args:
            other (Any): The value to compare with.

        Returns:
            BoolFeature: A boolean feature indicating whether the values are equal.
        """
        return self.execute_method("__eq__")

    def __ne__(self: PrimitiveFeature, other: Any) -> BoolFeature:
        """Compares if the feature is not equal to another value.

        Args:
            other (Any): The value to compare with.

        Returns:
            BoolFeature: A boolean feature indicating whether the values are not equal.
        """
        return self.execute_method("__ne__")

    def __lt__(self: PrimitiveFeature, other: Any) -> BoolFeature:
        """Compares if the feature is less than another value.

        Args:
            other (Any): The value to compare with.

        Returns:
            BoolFeature: A boolean feature indicating whether the feature is less than the given
            value.
        """
        return self.execute_method("__lt__")

    def __le__(self: PrimitiveFeature, other: Any) -> BoolFeature:
        """Compares if the feature is less than or equal to another value.

        Args:
            other (Any): The value to compare with.

        Returns:
            BoolFeature: A boolean feature indicating whether the feature is less than or equal
            to the given value.
        """
        return self.execute_method("__le__")

    def __gt__(self: PrimitiveFeature, other: Any) -> BoolFeature:
        """Compares if the feature is greater than another value.

        Args:
            other (Any): The value to compare with.

        Returns:
            BoolFeature: A boolean feature indicating whether the feature is greater than the
            given value.
        """
        return self.execute_method("__gt__")

    def __ge__(self: PrimitiveFeature, other: Any) -> BoolFeature:
        """Compares if the feature is greater than or equal to another value.

        Args:
            other (Any): The value to compare with.

        Returns:
            BoolFeature: A boolean feature indicating whether the feature is greater than or equal
            to the given value.
        """
        return self.execute_method("__ge__")


class Int8Feature(
    PrimitiveFeature[dtypes.Int8Type], ComparableFeatureMixin, StatisticalFeatureMixin
):
    """A primitive feature representing a signed 8-bit integer."""

    _expected_dtype: Final[dtypes.Type] = dtypes.Int8Type

    def __abs__(self) -> Int8Feature:
        """Performs the absolute value operation ('abs').

        Returns:
            Int8Feature: A new feature representing the absolute value of the original feature.
        """
        return self.execute_method("__abs__")

    def __neg__(self) -> Int8Feature:
        """Performs the negation operation ('-').

        Returns:
            Int8Feature: A new feature representing the negated value of the original feature.
        """
        return self.execute_method("__neg__")

    @overload
    def __add__(self, other: Int8Feature) -> Int8Feature:
        ...

    @overload
    def __add__(self, other: Int16Feature) -> Int16Feature:
        ...

    @overload
    def __add__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __add__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: UInt8Feature) -> Int16Feature:
        ...

    @overload
    def __add__(self, other: UInt16Feature) -> Int32Feature:
        ...

    @overload
    def __add__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __add__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __add__(self, other: int) -> Int8Feature:
        ...

    @overload
    def __add__(self, other: float) -> Float64Feature:
        ...

    def __add__(self, other: Any) -> Feature:
        """Performs addition operation ('+').

        Returns:
            Feature: A new feature representing the sum of the original features.
        """
        return self.execute_method("__add__", other)

    @overload
    def __radd__(self, other: int) -> Int8Feature:
        ...

    @overload
    def __radd__(self, other: float) -> Float64Feature:
        ...

    def __radd__(self, other: Any) -> Feature:
        """Performs right addition operation with constant ('+').

        Returns:
            Feature: A new feature representing the sum of the original feature and the constant.
        """
        return type(self).get_method("__add__")(other, self)

    @overload
    def __sub__(self, other: Int8Feature) -> Int8Feature:
        ...

    @overload
    def __sub__(self, other: Int16Feature) -> Int16Feature:
        ...

    @overload
    def __sub__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __sub__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: UInt8Feature) -> Int16Feature:
        ...

    @overload
    def __sub__(self, other: UInt16Feature) -> Int32Feature:
        ...

    @overload
    def __sub__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __sub__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __sub__(self, other: int) -> Int8Feature:
        ...

    @overload
    def __sub__(self, other: float) -> Float64Feature:
        ...

    def __sub__(self, other: Any) -> Feature:
        """Performs subtraction operation ('-').

        Returns:
            Feature: A new feature representing the difference of the original features.
        """
        return self.execute_method("__sub__", other)

    @overload
    def __rsub__(self, other: int) -> Int8Feature:
        ...

    @overload
    def __rsub__(self, other: float) -> Float64Feature:
        ...

    def __rsub__(self, other: Any) -> Feature:
        """Performs right subtraction operation with constant ('-').

        Returns:
            Feature: A new feature representing the difference between the constant and the
            original feature.
        """
        return type(self).get_method("__sub__")(other, self)

    @overload
    def __mul__(self, other: Int8Feature) -> Int8Feature:
        ...

    @overload
    def __mul__(self, other: Int16Feature) -> Int16Feature:
        ...

    @overload
    def __mul__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __mul__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: UInt8Feature) -> Int16Feature:
        ...

    @overload
    def __mul__(self, other: UInt16Feature) -> Int32Feature:
        ...

    @overload
    def __mul__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __mul__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __mul__(self, other: int) -> Int8Feature:
        ...

    @overload
    def __mul__(self, other: float) -> Float64Feature:
        ...

    def __mul__(self, other: Any) -> Feature:
        """Performs multiplication operation ('*').

        Returns:
            Feature: A new feature representing the product of the original features.
        """
        return self.execute_method("__mul__", other)

    @overload
    def __rmul__(self, other: int) -> Int8Feature:
        ...

    @overload
    def __rmul__(self, other: float) -> Float64Feature:
        ...

    def __rmul__(self, other: Any) -> Feature:
        """Performs right multiplication operation with constant ('*').

        Returns:
            Feature: A new feature representing the product of the constant and the
            original feature.
        """
        return type(self).get_method("__mul__")(other, self)

    @overload
    def __truediv__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
        ),
    ) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __truediv__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: float) -> Float64Feature:
        ...

    def __truediv__(self, other: Any) -> Feature:
        """Performs true division operation ('/').

        Returns:
            Feature: A new feature representing the quotient of the original features.
        """
        return self.execute_method("__truediv__", other)

    @overload
    def __rtruediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __rtruediv__(self, other: float) -> Float64Feature:
        ...

    def __rtruediv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__truediv__")(other, self)

    @overload
    def __floordiv__(self, other: Int8Feature) -> Int8Feature:
        ...

    @overload
    def __floordiv__(self, other: Int16Feature) -> Int16Feature:
        ...

    @overload
    def __floordiv__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt8Feature) -> Int16Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt16Feature) -> Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: int) -> Int8Feature:
        ...

    @overload
    def __floordiv__(self, other: float) -> Int64Feature:
        ...

    def __floordiv__(self, other: Any) -> Feature:
        """Performs floor division operation ('//').

        Returns:
            Feature: A new feature representing the integer quotient of the original features.
        """
        return self.execute_method("__floordiv__", other)

    @overload
    def __rfloordiv__(self, other: int) -> Int8Feature:
        ...

    @overload
    def __rfloordiv__(self, other: float) -> Int64Feature:
        ...

    def __rfloordiv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__floordiv__")(other, self)


class Int16Feature(
    PrimitiveFeature[dtypes.Int16Type], ComparableFeatureMixin, StatisticalFeatureMixin
):
    """A primitive feature representing a signed 16-bit integer."""

    _expected_dtype: Final[dtypes.Type] = dtypes.Int16Type

    def __abs__(self) -> Int16Feature:
        """Performs the absolute value operation ('abs').

        Returns:
            Int16Feature: A new feature representing the absolute value of the original feature.
        """
        return self.execute_method("__abs__")

    def __neg__(self) -> Int16Feature:
        """Performs the negation operation ('-').

        Returns:
            Int16Feature: A new feature representing the negated value of the original feature.
        """
        return self.execute_method("__neg__")

    @overload
    def __add__(self, other: Int8Feature) -> Int16Feature:
        ...

    @overload
    def __add__(self, other: Int16Feature) -> Int16Feature:
        ...

    @overload
    def __add__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __add__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: UInt8Feature) -> Int16Feature:
        ...

    @overload
    def __add__(self, other: UInt16Feature) -> Int32Feature:
        ...

    @overload
    def __add__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __add__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __add__(self, other: int) -> Int16Feature:
        ...

    @overload
    def __add__(self, other: float) -> Float64Feature:
        ...

    def __add__(self, other: Any) -> Feature:
        """Performs addition operation ('+').

        Returns:
            Feature: A new feature representing the sum of the original features.
        """
        return self.execute_method("__add__", other)

    @overload
    def __radd__(self, other: int) -> Int16Feature:
        ...

    @overload
    def __radd__(self, other: float) -> Float64Feature:
        ...

    def __radd__(self, other: Any) -> Feature:
        """Performs right addition operation with constant ('+').

        Returns:
            Feature: A new feature representing the sum of the original feature and the constant.
        """
        return type(self).get_method("__add__")(other, self)

    @overload
    def __sub__(self, other: Int8Feature) -> Int16Feature:
        ...

    @overload
    def __sub__(self, other: Int16Feature) -> Int16Feature:
        ...

    @overload
    def __sub__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __sub__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: UInt8Feature) -> Int16Feature:
        ...

    @overload
    def __sub__(self, other: UInt16Feature) -> Int32Feature:
        ...

    @overload
    def __sub__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __sub__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __sub__(self, other: int) -> Int16Feature:
        ...

    @overload
    def __sub__(self, other: float) -> Float64Feature:
        ...

    def __sub__(self, other: Any) -> Feature:
        """Performs subtraction operation ('-').

        Returns:
            Feature: A new feature representing the difference of the original features.
        """
        return self.execute_method("__sub__", other)

    @overload
    def __rsub__(self, other: int) -> Int16Feature:
        ...

    @overload
    def __rsub__(self, other: float) -> Float64Feature:
        ...

    def __rsub__(self, other: Any) -> Feature:
        """Performs right subtraction operation with constant ('-').

        Returns:
            Feature: A new feature representing the difference between the constant and the
            original feature.
        """
        return type(self).get_method("__sub__")(other, self)

    @overload
    def __mul__(self, other: Int8Feature) -> Int16Feature:
        ...

    @overload
    def __mul__(self, other: Int16Feature) -> Int16Feature:
        ...

    @overload
    def __mul__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __mul__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: UInt8Feature) -> Int16Feature:
        ...

    @overload
    def __mul__(self, other: UInt16Feature) -> Int32Feature:
        ...

    @overload
    def __mul__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __mul__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __mul__(self, other: int) -> Int16Feature:
        ...

    @overload
    def __mul__(self, other: float) -> Float64Feature:
        ...

    def __mul__(self, other: Any) -> Feature:
        """Performs multiplication operation ('*').

        Returns:
            Feature: A new feature representing the product of the original features.
        """
        return self.execute_method("__mul__", other)

    @overload
    def __rmul__(self, other: int) -> Int16Feature:
        ...

    @overload
    def __rmul__(self, other: float) -> Float64Feature:
        ...

    def __rmul__(self, other: Any) -> Feature:
        """Performs right multiplication operation with constant ('*').

        Returns:
            Feature: A new feature representing the product of the constant and the
            original feature.
        """
        return type(self).get_method("__mul__")(other, self)

    @overload
    def __truediv__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
        ),
    ) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __truediv__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: float) -> Float64Feature:
        ...

    def __truediv__(self, other: Any) -> Feature:
        """Performs true division operation ('/').

        Returns:
            Feature: A new feature representing the quotient of the original features.
        """
        return self.execute_method("__truediv__", other)

    @overload
    def __rtruediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __rtruediv__(self, other: float) -> Float64Feature:
        ...

    def __rtruediv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__truediv__")(other, self)

    @overload
    def __floordiv__(self, other: Int8Feature) -> Int16Feature:
        ...

    @overload
    def __floordiv__(self, other: Int16Feature) -> Int16Feature:
        ...

    @overload
    def __floordiv__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt8Feature) -> Int16Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt16Feature) -> Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: int) -> Int16Feature:
        ...

    @overload
    def __floordiv__(self, other: float) -> Int64Feature:
        ...

    def __floordiv__(self, other: Any) -> Feature:
        """Performs floor division operation ('//').

        Returns:
            Feature: A new feature representing the integer quotient of the original features.
        """
        return self.execute_method("__floordiv__", other)

    @overload
    def __rfloordiv__(self, other: int) -> Int16Feature:
        ...

    @overload
    def __rfloordiv__(self, other: float) -> Int64Feature:
        ...

    def __rfloordiv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__floordiv__")(other, self)


class Int32Feature(
    PrimitiveFeature[dtypes.Int32Type], ComparableFeatureMixin, StatisticalFeatureMixin
):
    """A primitive feature representing a signed 32-bit integer."""

    _expected_dtype: Final[dtypes.Type] = dtypes.Int32Type

    def __abs__(self) -> Int32Feature:
        """Performs the absolute value operation ('abs').

        Returns:
            Int32Feature: A new feature representing the absolute value of the original feature.
        """
        return self.execute_method("__abs__")

    def __neg__(self) -> Int32Feature:
        """Performs the negation operation ('-').

        Returns:
            Int32Feature: A new feature representing the negated value of the original feature.
        """
        return self.execute_method("__neg__")

    @overload
    def __add__(self, other: Int8Feature) -> Int32Feature:
        ...

    @overload
    def __add__(self, other: Int16Feature) -> Int32Feature:
        ...

    @overload
    def __add__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __add__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: UInt8Feature) -> Int32Feature:
        ...

    @overload
    def __add__(self, other: UInt16Feature) -> Int32Feature:
        ...

    @overload
    def __add__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __add__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __add__(self, other: int) -> Int32Feature:
        ...

    @overload
    def __add__(self, other: float) -> Float64Feature:
        ...

    def __add__(self, other: Any) -> Feature:
        """Performs addition operation ('+').

        Returns:
            Feature: A new feature representing the sum of the original features.
        """
        return self.execute_method("__add__", other)

    @overload
    def __radd__(self, other: int) -> Int32Feature:
        ...

    @overload
    def __radd__(self, other: float) -> Float64Feature:
        ...

    def __radd__(self, other: Any) -> Feature:
        """Performs right addition operation with constant ('+').

        Returns:
            Feature: A new feature representing the sum of the original feature and the constant.
        """
        return type(self).get_method("__add__")(other, self)

    @overload
    def __sub__(self, other: Int8Feature) -> Int32Feature:
        ...

    @overload
    def __sub__(self, other: Int16Feature) -> Int32Feature:
        ...

    @overload
    def __sub__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __sub__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: UInt8Feature) -> Int32Feature:
        ...

    @overload
    def __sub__(self, other: UInt16Feature) -> Int32Feature:
        ...

    @overload
    def __sub__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __sub__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __sub__(self, other: int) -> Int32Feature:
        ...

    @overload
    def __sub__(self, other: float) -> Float64Feature:
        ...

    def __sub__(self, other: Any) -> Feature:
        """Performs subtraction operation ('-').

        Returns:
            Feature: A new feature representing the difference of the original features.
        """
        return self.execute_method("__sub__", other)

    @overload
    def __rsub__(self, other: int) -> Int32Feature:
        ...

    @overload
    def __rsub__(self, other: float) -> Float64Feature:
        ...

    def __rsub__(self, other: Any) -> Feature:
        """Performs right subtraction operation with constant ('-').

        Returns:
            Feature: A new feature representing the difference between the constant and the
            original feature.
        """
        return type(self).get_method("__sub__")(other, self)

    @overload
    def __mul__(self, other: Int8Feature) -> Int32Feature:
        ...

    @overload
    def __mul__(self, other: Int16Feature) -> Int32Feature:
        ...

    @overload
    def __mul__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __mul__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: UInt8Feature) -> Int32Feature:
        ...

    @overload
    def __mul__(self, other: UInt16Feature) -> Int32Feature:
        ...

    @overload
    def __mul__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __mul__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __mul__(self, other: int) -> Int32Feature:
        ...

    @overload
    def __mul__(self, other: float) -> Float64Feature:
        ...

    def __mul__(self, other: Any) -> Feature:
        """Performs multiplication operation ('*').

        Returns:
            Feature: A new feature representing the product of the original features.
        """
        return self.execute_method("__mul__", other)

    @overload
    def __rmul__(self, other: int) -> Int32Feature:
        ...

    @overload
    def __rmul__(self, other: float) -> Float64Feature:
        ...

    def __rmul__(self, other: Any) -> Feature:
        """Performs right multiplication operation with constant ('*').

        Returns:
            Feature: A new feature representing the product of the constant and the
            original feature.
        """
        return type(self).get_method("__mul__")(other, self)

    @overload
    def __truediv__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
        ),
    ) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __truediv__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: float) -> Float64Feature:
        ...

    def __truediv__(self, other: Any) -> Feature:
        """Performs true division operation ('/').

        Returns:
            Feature: A new feature representing the quotient of the original features.
        """
        return self.execute_method("__truediv__", other)

    @overload
    def __rtruediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __rtruediv__(self, other: float) -> Float64Feature:
        ...

    def __rtruediv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__truediv__")(other, self)

    @overload
    def __floordiv__(self, other: Int8Feature) -> Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: Int16Feature) -> Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt8Feature) -> Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt16Feature) -> Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: int) -> Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: float) -> Int64Feature:
        ...

    def __floordiv__(self, other: Any) -> Feature:
        """Performs floor division operation ('//').

        Returns:
            Feature: A new feature representing the integer quotient of the original features.
        """
        return self.execute_method("__floordiv__", other)

    @overload
    def __rfloordiv__(self, other: int) -> Int32Feature:
        ...

    @overload
    def __rfloordiv__(
        self, other: float
    ) -> Int64Feature:  # TODO: why is this 64-bit, is it really?
        ...

    def __rfloordiv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__floordiv__")(other, self)


class Int64Feature(
    PrimitiveFeature[dtypes.Int64Type], ComparableFeatureMixin, StatisticalFeatureMixin
):
    """A primitive feature representing a signed 64-bit integer."""

    _expected_dtype: Final[dtypes.Type] = dtypes.Int64Type

    def __abs__(self) -> Int64Feature:
        """Performs the absolute value operation ('abs').

        Returns:
            Int64Feature: A new feature representing the absolute value of the original feature.
        """
        return self.execute_method("__abs__")

    def __neg__(self) -> Int64Feature:
        """Performs the negation operation ('-').

        Returns:
            Int64Feature: A new feature representing the negated value of the original feature.
        """
        return self.execute_method("__neg__")

    @overload
    def __add__(self, other: Int8Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: Int16Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: Int32Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: UInt8Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: UInt16Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __add__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __add__(self, other: int) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: float) -> Float64Feature:
        ...

    def __add__(self, other: Any) -> Feature:
        """Performs addition operation ('+').

        Returns:
            Feature: A new feature representing the sum of the original features.
        """
        return self.execute_method("__add__", other)

    @overload
    def __radd__(self, other: int) -> Int64Feature:
        ...

    @overload
    def __radd__(self, other: float) -> Float64Feature:
        ...

    def __radd__(self, other: Any) -> Feature:
        """Performs right addition operation with constant ('+').

        Returns:
            Feature: A new feature representing the sum of the original feature and the constant.
        """
        return type(self).get_method("__add__")(other, self)

    @overload
    def __sub__(self, other: Int8Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: Int16Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: Int32Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: UInt8Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: UInt16Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __sub__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __sub__(self, other: int) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: float) -> Float64Feature:
        ...

    def __sub__(self, other: Any) -> Feature:
        """Performs subtraction operation ('-').

        Returns:
            Feature: A new feature representing the difference of the original features.
        """
        return self.execute_method("__sub__", other)

    @overload
    def __rsub__(self, other: int) -> Int64Feature:
        ...

    @overload
    def __rsub__(self, other: float) -> Float64Feature:
        ...

    def __rsub__(self, other: Any) -> Feature:
        """Performs right subtraction operation with constant ('-').

        Returns:
            Feature: A new feature representing the difference between the constant and the
            original feature.
        """
        return type(self).get_method("__sub__")(other, self)

    @overload
    def __mul__(self, other: Int8Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: Int16Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: Int32Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: UInt8Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: UInt16Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __mul__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __mul__(self, other: int) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: float) -> Float64Feature:
        ...

    def __mul__(self, other: Any) -> Feature:
        """Performs multiplication operation ('*').

        Returns:
            Feature: A new feature representing the product of the original features.
        """
        return self.execute_method("__mul__", other)

    @overload
    def __rmul__(self, other: int) -> Int64Feature:
        ...

    @overload
    def __rmul__(self, other: float) -> Float64Feature:
        ...

    def __rmul__(self, other: Any) -> Feature:
        """Performs right multiplication operation with constant ('*').

        Returns:
            Feature: A new feature representing the product of the constant and the
            original feature.
        """
        return type(self).get_method("__mul__")(other, self)

    @overload
    def __truediv__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
        ),
    ) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __truediv__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: float) -> Float64Feature:
        ...

    def __truediv__(self, other: Any) -> Feature:
        """Performs true division operation ('/').

        Returns:
            Feature: A new feature representing the quotient of the original features.
        """
        return self.execute_method("__truediv__", other)

    @overload
    def __rtruediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __rtruediv__(self, other: float) -> Float64Feature:
        ...

    def __rtruediv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__truediv__")(other, self)

    @overload
    def __floordiv__(self, other: Int8Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Int16Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Int32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt8Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt16Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: int) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: float) -> Int64Feature:
        ...

    def __floordiv__(self, other: Any) -> Feature:
        """Performs floor division operation ('//').

        Returns:
            Feature: A new feature representing the integer quotient of the original features.
        """
        return self.execute_method("__floordiv__", other)

    @overload
    def __rfloordiv__(self, other: int) -> Int64Feature:
        ...

    @overload
    def __rfloordiv__(self, other: float) -> Int64Feature:
        ...

    def __rfloordiv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__floordiv__")(other, self)


class UInt8Feature(
    PrimitiveFeature[dtypes.UInt8Type], ComparableFeatureMixin, StatisticalFeatureMixin
):
    """A primitive feature representing an unsigned 8-bit integer."""

    _expected_dtype: Final[dtypes.Type] = dtypes.UInt8Type

    def __abs__(self) -> UInt8Feature:
        """Performs the absolute value operation ('abs').

        Returns:
            UInt8Feature: A new feature representing the absolute value of the original feature.
        """
        return self.execute_method("__abs__")

    def __neg__(self: Feature) -> Int16Feature:
        """Performs the negation operation ('-').

        Returns:
            Int16Feature: A new feature representing the negated value of the original feature.
        """
        return self.execute_method("__neg__")

    @overload
    def __add__(self, other: Int8Feature) -> Int16Feature:
        ...

    @overload
    def __add__(self, other: Int16Feature) -> Int16Feature:
        ...

    @overload
    def __add__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __add__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: UInt8Feature) -> UInt8Feature:
        ...

    @overload
    def __add__(self, other: UInt16Feature) -> UInt16Feature:
        ...

    @overload
    def __add__(self, other: UInt32Feature) -> UInt32Feature:
        ...

    @overload
    def __add__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __add__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __add__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __add__(self, other: int) -> UInt8Feature | Int32Feature:
        ...

    @overload
    def __add__(self, other: float) -> Float64Feature:
        ...

    def __add__(self, other: Any) -> Feature:
        """Performs addition operation ('+').

        Returns:
            Feature: A new feature representing the sum of the original features.
        """
        return self.execute_method("__add__", other)

    @overload
    def __radd__(self, other: int) -> UInt8Feature | Int32Feature:
        ...

    @overload
    def __radd__(self, other: float) -> Float64Feature:
        ...

    def __radd__(self, other: Any) -> Feature:
        """Performs right addition operation with constant ('+').

        Returns:
            Feature: A new feature representing the sum of the original feature and the constant.
        """
        return type(self).get_method("__add__")(other, self)

    @overload
    def __sub__(self, other: Int8Feature) -> Int16Feature:
        ...

    @overload
    def __sub__(self, other: Int16Feature) -> Int16Feature:
        ...

    @overload
    def __sub__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __sub__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: UInt8Feature) -> UInt8Feature:
        ...

    @overload
    def __sub__(self, other: UInt16Feature) -> UInt16Feature:
        ...

    @overload
    def __sub__(self, other: UInt32Feature) -> UInt32Feature:
        ...

    @overload
    def __sub__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __sub__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __sub__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __sub__(self, other: int) -> UInt8Feature | Int32Feature:
        ...

    @overload
    def __sub__(self, other: float) -> Float64Feature:
        ...

    def __sub__(self, other: Any) -> Feature:
        """Performs subtraction operation ('-').

        Returns:
            Feature: A new feature representing the difference of the original features.
        """
        return self.execute_method("__sub__", other)

    @overload
    def __rsub__(self, other: int) -> UInt8Feature | Int32Feature:
        ...

    @overload
    def __rsub__(self, other: float) -> Float64Feature:
        ...

    def __rsub__(self, other: Any) -> Feature:
        """Performs right subtraction operation with constant ('-').

        Returns:
            Feature: A new feature representing the difference between the constant and the
            original feature.
        """
        return type(self).get_method("__sub__")(other, self)

    @overload
    def __mul__(self, other: Int8Feature) -> Int16Feature:
        ...

    @overload
    def __mul__(self, other: Int16Feature) -> Int16Feature:
        ...

    @overload
    def __mul__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __mul__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: UInt8Feature) -> UInt8Feature:
        ...

    @overload
    def __mul__(self, other: UInt16Feature) -> UInt16Feature:
        ...

    @overload
    def __mul__(self, other: UInt32Feature) -> UInt32Feature:
        ...

    @overload
    def __mul__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __mul__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __mul__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __mul__(self, other: int) -> UInt8Feature | Int32Feature:
        ...

    @overload
    def __mul__(self, other: float) -> Float64Feature:
        ...

    def __mul__(self, other: Any) -> Feature:
        """Performs multiplication operation ('*').

        Returns:
            Feature: A new feature representing the product of the original features.
        """
        return self.execute_method("__mul__", other)

    @overload
    def __rmul__(self, other: int) -> UInt8Feature | Int32Feature:
        ...

    @overload
    def __rmul__(self, other: float) -> Float64Feature:
        ...

    def __rmul__(self, other: Any) -> Feature:
        """Performs right multiplication operation with constant ('*').

        Returns:
            Feature: A new feature representing the product of the constant and the
            original feature.
        """
        return type(self).get_method("__mul__")(other, self)

    @overload
    def __truediv__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
        ),
    ) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __truediv__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: float) -> Float64Feature:
        ...

    def __truediv__(self, other: Any) -> Feature:
        """Performs true division operation ('/').

        Returns:
            Feature: A new feature representing the quotient of the original features.
        """
        return self.execute_method("__truediv__", other)

    @overload
    def __rtruediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __rtruediv__(self, other: float) -> Float64Feature:
        ...

    def __rtruediv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__truediv__")(other, self)

    @overload
    def __floordiv__(self, other: Int8Feature) -> Int16Feature:
        ...

    @overload
    def __floordiv__(self, other: Int16Feature) -> Int16Feature:
        ...

    @overload
    def __floordiv__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt8Feature) -> UInt8Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt16Feature) -> UInt16Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt32Feature) -> UInt32Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: int) -> UInt8Feature | Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: float) -> Int64Feature:
        ...

    def __floordiv__(self, other: Any) -> Feature:
        """Performs floor division operation ('//').

        Returns:
            Feature: A new feature representing the integer quotient of the original features.
        """
        return self.execute_method("__floordiv__", other)

    @overload
    def __rfloordiv__(self, other: int) -> UInt8Feature | Int32Feature:
        ...

    @overload
    def __rfloordiv__(self, other: float) -> Int64Feature:
        ...

    def __rfloordiv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__floordiv__")(other, self)


class UInt16Feature(
    PrimitiveFeature[dtypes.UInt16Type], ComparableFeatureMixin, StatisticalFeatureMixin
):
    """A primitive feature representing an unsigned 16-bit integer."""

    _expected_dtype: Final[dtypes.Type] = dtypes.UInt16Type

    def __abs__(self) -> UInt16Feature:
        """Performs the absolute value operation ('abs').

        Returns:
            UInt16Feature: A new feature representing the absolute value of the original feature.
        """
        return self.execute_method("__abs__")

    def __neg__(self: Feature) -> Int32Feature:
        """Performs the negation operation ('-').

        Returns:
            Int32Feature: A new feature representing the negated value of the original feature.
        """
        return self.execute_method("__neg__")

    @overload
    def __add__(self, other: Int8Feature) -> Int32Feature:
        ...

    @overload
    def __add__(self, other: Int16Feature) -> Int32Feature:
        ...

    @overload
    def __add__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __add__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: UInt8Feature) -> UInt16Feature:
        ...

    @overload
    def __add__(self, other: UInt16Feature) -> UInt16Feature:
        ...

    @overload
    def __add__(self, other: UInt32Feature) -> UInt32Feature:
        ...

    @overload
    def __add__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __add__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __add__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __add__(self, other: int) -> UInt16Feature | Int32Feature:
        ...

    @overload
    def __add__(self, other: float) -> Float64Feature:
        ...

    def __add__(self, other: Any) -> Feature:
        """Performs addition operation ('+').

        Returns:
            Feature: A new feature representing the sum of the original features.
        """
        return self.execute_method("__add__", other)

    @overload
    def __radd__(self, other: int) -> UInt16Feature | Int32Feature:
        ...

    @overload
    def __radd__(self, other: float) -> Float64Feature:
        ...

    def __radd__(self, other: Any) -> Feature:
        """Performs right addition operation with constant ('+').

        Returns:
            Feature: A new feature representing the sum of the original feature and the constant.
        """
        return type(self).get_method("__add__")(other, self)

    @overload
    def __sub__(self, other: Int8Feature) -> Int32Feature:
        ...

    @overload
    def __sub__(self, other: Int16Feature) -> Int32Feature:
        ...

    @overload
    def __sub__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __sub__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: UInt8Feature) -> UInt16Feature:
        ...

    @overload
    def __sub__(self, other: UInt16Feature) -> UInt16Feature:
        ...

    @overload
    def __sub__(self, other: UInt32Feature) -> UInt32Feature:
        ...

    @overload
    def __sub__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __sub__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __sub__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __sub__(self, other: int) -> UInt16Feature | Int32Feature:
        ...

    @overload
    def __sub__(self, other: float) -> Float64Feature:
        ...

    def __sub__(self, other: Any) -> Feature:
        """Performs subtraction operation ('-').

        Returns:
            Feature: A new feature representing the difference of the original features.
        """
        return self.execute_method("__sub__", other)

    @overload
    def __rsub__(self, other: int) -> UInt16Feature | Int32Feature:
        ...

    @overload
    def __rsub__(self, other: float) -> Float64Feature:
        ...

    def __rsub__(self, other: Any) -> Feature:
        """Performs right subtraction operation with constant ('-').

        Returns:
            Feature: A new feature representing the difference between the constant and the
            original feature.
        """
        return type(self).get_method("__sub__")(other, self)

    @overload
    def __mul__(self, other: Int8Feature) -> Int32Feature:
        ...

    @overload
    def __mul__(self, other: Int16Feature) -> Int32Feature:
        ...

    @overload
    def __mul__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __mul__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: UInt8Feature) -> UInt16Feature:
        ...

    @overload
    def __mul__(self, other: UInt16Feature) -> UInt16Feature:
        ...

    @overload
    def __mul__(self, other: UInt32Feature) -> UInt32Feature:
        ...

    @overload
    def __mul__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __mul__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __mul__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __mul__(self, other: int) -> UInt16Feature | Int32Feature:
        ...

    @overload
    def __mul__(self, other: float) -> Float64Feature:
        ...

    def __mul__(self, other: Any) -> Feature:
        """Performs multiplication operation ('*').

        Returns:
            Feature: A new feature representing the product of the original features.
        """
        return self.execute_method("__mul__", other)

    @overload
    def __rmul__(self, other: int) -> UInt16Feature | Int32Feature:
        ...

    @overload
    def __rmul__(self, other: float) -> Float64Feature:
        ...

    def __rmul__(self, other: Any) -> Feature:
        """Performs right multiplication operation with constant ('*').

        Returns:
            Feature: A new feature representing the product of the constant and the
            original feature.
        """
        return type(self).get_method("__mul__")(other, self)

    @overload
    def __truediv__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
        ),
    ) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __truediv__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: float) -> Float64Feature:
        ...

    def __truediv__(self, other: Any) -> Feature:
        """Performs true division operation ('/').

        Returns:
            Feature: A new feature representing the quotient of the original features.
        """
        return self.execute_method("__truediv__", other)

    @overload
    def __rtruediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __rtruediv__(self, other: float) -> Float64Feature:
        ...

    def __rtruediv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__truediv__")(other, self)

    @overload
    def __floordiv__(self, other: Int8Feature) -> Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: Int16Feature) -> Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: Int32Feature) -> Int32Feature:
        ...

    @overload
    def __floordiv__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt8Feature) -> UInt16Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt16Feature) -> UInt16Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt32Feature) -> UInt32Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: int) -> Int32Feature | UInt16Feature:
        ...

    @overload
    def __floordiv__(self, other: float) -> Int64Feature:
        ...

    def __floordiv__(self, other: Any) -> Feature:
        """Performs floor division operation ('//').

        Returns:
            Feature: A new feature representing the integer quotient of the original features.
        """
        return self.execute_method("__floordiv__", other)

    @overload
    def __rfloordiv__(self, other: int) -> Int32Feature | UInt16Feature:
        ...

    @overload
    def __rfloordiv__(self, other: float) -> Int64Feature:
        ...

    def __rfloordiv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__floordiv__")(other, self)


class UInt32Feature(
    PrimitiveFeature[dtypes.UInt32Type], ComparableFeatureMixin, StatisticalFeatureMixin
):
    """A primitive feature representing an unsigned 32-bit integer."""

    _expected_dtype: Final[dtypes.Type] = dtypes.UInt32Type

    def __abs__(self) -> UInt32Feature:
        """Performs the absolute value operation ('abs').

        Returns:
            UInt32Feature: A new feature representing the absolute value of the original feature.
        """
        return self.execute_method("__abs__")

    def __neg__(self: Feature) -> Int64Feature:
        """Performs the negation operation ('-').

        Returns:
            Int64Feature: A new feature representing the negated value of the original feature.
        """
        return self.execute_method("__neg__")

    @overload
    def __add__(self, other: Int8Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: Int16Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: Int32Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: UInt8Feature) -> UInt32Feature:
        ...

    @overload
    def __add__(self, other: UInt16Feature) -> UInt32Feature:
        ...

    @overload
    def __add__(self, other: UInt32Feature) -> UInt32Feature:
        ...

    @overload
    def __add__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __add__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __add__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __add__(self, other: int) -> UInt32Feature | Int64Feature:
        ...

    @overload
    def __add__(self, other: float) -> Float64Feature:
        ...

    def __add__(self, other: Any) -> Feature:
        """Performs addition operation ('+').

        Returns:
            Feature: A new feature representing the sum of the original features.
        """
        return self.execute_method("__add__", other)

    @overload
    def __radd__(self, other: int) -> UInt32Feature | Int64Feature:
        ...

    @overload
    def __radd__(self, other: float) -> Float64Feature:
        ...

    def __radd__(self, other: Any) -> Feature:
        """Performs right addition operation with constant ('+').

        Returns:
            Feature: A new feature representing the sum of the original feature and the constant.
        """
        return type(self).get_method("__add__")(other, self)

    @overload
    def __sub__(self, other: Int8Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: Int16Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: Int32Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: UInt8Feature) -> UInt32Feature:
        ...

    @overload
    def __sub__(self, other: UInt16Feature) -> UInt32Feature:
        ...

    @overload
    def __sub__(self, other: UInt32Feature) -> UInt32Feature:
        ...

    @overload
    def __sub__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __sub__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __sub__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __sub__(self, other: int) -> UInt32Feature | Int64Feature:
        ...

    @overload
    def __sub__(self, other: float) -> Float64Feature:
        ...

    def __sub__(self, other: Any) -> Feature:
        """Performs subtraction operation ('-').

        Returns:
            Feature: A new feature representing the difference of the original features.
        """
        return self.execute_method("__sub__", other)

    @overload
    def __rsub__(self, other: int) -> UInt32Feature | Int64Feature:
        ...

    @overload
    def __rsub__(self, other: float) -> Float64Feature:
        ...

    def __rsub__(self, other: Any) -> Feature:
        """Performs right subtraction operation with constant ('-').

        Returns:
            Feature: A new feature representing the difference between the constant and the
            original feature.
        """
        return type(self).get_method("__sub__")(other, self)

    @overload
    def __mul__(self, other: Int8Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: Int16Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: Int32Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: UInt8Feature) -> UInt32Feature:
        ...

    @overload
    def __mul__(self, other: UInt16Feature) -> UInt32Feature:
        ...

    @overload
    def __mul__(self, other: UInt32Feature) -> UInt32Feature:
        ...

    @overload
    def __mul__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __mul__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __mul__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __mul__(self, other: int) -> UInt32Feature | Int64Feature:
        ...

    @overload
    def __mul__(self, other: float) -> Float64Feature:
        ...

    def __mul__(self, other: Any) -> Feature:
        """Performs multiplication operation ('*').

        Returns:
            Feature: A new feature representing the product of the original features.
        """
        return self.execute_method("__mul__", other)

    @overload
    def __rmul__(self, other: int) -> UInt32Feature | Int64Feature:
        ...

    @overload
    def __rmul__(self, other: float) -> Float64Feature:
        ...

    def __rmul__(self, other: Any) -> Feature:
        """Performs right multiplication operation with constant ('*').

        Returns:
            Feature: A new feature representing the product of the constant and the
            original feature.
        """
        return type(self).get_method("__mul__")(other, self)

    @overload
    def __truediv__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
        ),
    ) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __truediv__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: float) -> Float64Feature:
        ...

    def __truediv__(self, other: Any) -> Feature:
        """Performs true division operation ('/').

        Returns:
            Feature: A new feature representing the quotient of the original features.
        """
        return self.execute_method("__truediv__", other)

    @overload
    def __rtruediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __rtruediv__(self, other: float) -> Float64Feature:
        ...

    def __rtruediv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__truediv__")(other, self)

    @overload
    def __floordiv__(self, other: Int8Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Int16Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Int32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt8Feature) -> UInt32Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt16Feature) -> UInt32Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt32Feature) -> UInt32Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: int) -> Int64Feature | UInt32Feature:
        ...

    @overload
    def __floordiv__(self, other: float) -> Int64Feature:
        ...

    def __floordiv__(self, other: Any) -> Feature:
        """Performs floor division operation ('//').

        Returns:
            Feature: A new feature representing the integer quotient of the original features.
        """
        return self.execute_method("__floordiv__", other)

    @overload
    def __rfloordiv__(self, other: int) -> Int64Feature | UInt32Feature:
        ...

    @overload
    def __rfloordiv__(self, other: float) -> Int64Feature:
        ...

    def __rfloordiv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__floordiv__")(other, self)


class UInt64Feature(
    PrimitiveFeature[dtypes.UInt64Type], ComparableFeatureMixin, StatisticalFeatureMixin
):
    """A primitive feature representing an unsigned 64-bit integer."""

    _expected_dtype: Final[dtypes.Type] = dtypes.UInt64Type

    def __abs__(self) -> UInt64Feature:
        """Performs the absolute value operation ('abs').

        Returns:
            UInt64Feature: A new feature representing the absolute value of the original feature.
        """
        return self.execute_method("__abs__")

    def __neg__(self: Feature) -> Int64Feature:
        """Performs the negation operation ('-').

        Returns:
            Int64Feature: A new feature representing the negated value of the original feature.
        """
        return self.execute_method("__neg__")

    @overload
    def __add__(self, other: Int8Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: Int16Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: Int32Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __add__(self, other: UInt8Feature) -> UInt64Feature:
        ...

    @overload
    def __add__(self, other: UInt16Feature) -> UInt64Feature:
        ...

    @overload
    def __add__(self, other: UInt32Feature) -> UInt64Feature:
        ...

    @overload
    def __add__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __add__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __add__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __add__(self, other: int) -> UInt64Feature | Int64Feature:
        ...

    @overload
    def __add__(self, other: float) -> Float64Feature:
        ...

    def __add__(self, other: Any) -> Feature:
        """Performs addition operation ('+').

        Returns:
            Feature: A new feature representing the sum of the original features.
        """
        return self.execute_method("__add__", other)

    @overload
    def __radd__(self, other: int) -> UInt64Feature | Int64Feature:
        ...

    @overload
    def __radd__(self, other: float) -> Float64Feature:
        ...

    def __radd__(self, other: Any) -> Feature:
        """Performs right addition operation with constant ('+').

        Returns:
            Feature: A new feature representing the sum of the original feature and the constant.
        """
        return type(self).get_method("__add__")(other, self)

    @overload
    def __sub__(self, other: Int8Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: Int16Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: Int32Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __sub__(self, other: UInt8Feature) -> UInt64Feature:
        ...

    @overload
    def __sub__(self, other: UInt16Feature) -> UInt64Feature:
        ...

    @overload
    def __sub__(self, other: UInt32Feature) -> UInt64Feature:
        ...

    @overload
    def __sub__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __sub__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __sub__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __sub__(self, other: int) -> UInt64Feature | Int64Feature:
        ...

    @overload
    def __sub__(self, other: float) -> Float64Feature:
        ...

    def __sub__(self, other: Any) -> Feature:
        """Performs subtraction operation ('-').

        Returns:
            Feature: A new feature representing the difference of the original features.
        """
        return self.execute_method("__sub__", other)

    @overload
    def __rsub__(self, other: int) -> UInt64Feature | Int64Feature:
        ...

    @overload
    def __rsub__(self, other: float) -> Float64Feature:
        ...

    def __rsub__(self, other: Any) -> Feature:
        """Performs right subtraction operation with constant ('-').

        Returns:
            Feature: A new feature representing the difference between the constant and the
            original feature.
        """
        return type(self).get_method("__sub__")(other, self)

    @overload
    def __mul__(self, other: Int8Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: Int16Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: Int32Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __mul__(self, other: UInt8Feature) -> UInt64Feature:
        ...

    @overload
    def __mul__(self, other: UInt16Feature) -> UInt64Feature:
        ...

    @overload
    def __mul__(self, other: UInt32Feature) -> UInt64Feature:
        ...

    @overload
    def __mul__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __mul__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __mul__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __mul__(self, other: int) -> UInt64Feature | Int64Feature:
        ...

    @overload
    def __mul__(self, other: float) -> Float64Feature:
        ...

    def __mul__(self, other: Any) -> Feature:
        """Performs multiplication operation ('*').

        Returns:
            Feature: A new feature representing the product of the original features.
        """
        return self.execute_method("__mul__", other)

    @overload
    def __rmul__(self, other: int) -> UInt64Feature | Int64Feature:
        ...

    @overload
    def __rmul__(self, other: float) -> Float64Feature:
        ...

    def __rmul__(self, other: Any) -> Feature:
        """Performs right multiplication operation with constant ('*').

        Returns:
            Feature: A new feature representing the product of the constant and the
            original feature.
        """
        return type(self).get_method("__mul__")(other, self)

    @overload
    def __truediv__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
        ),
    ) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __truediv__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: float) -> Float64Feature:
        ...

    def __truediv__(self, other: Any) -> Feature:
        """Performs true division operation ('/').

        Returns:
            Feature: A new feature representing the quotient of the original features.
        """
        return self.execute_method("__truediv__", other)

    @overload
    def __rtruediv__(self, other: int) -> Float64Feature:
        ...

    @overload
    def __rtruediv__(self, other: float) -> Float64Feature:
        ...

    def __rtruediv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__truediv__")(other, self)

    @overload
    def __floordiv__(self, other: Int8Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Int16Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Int32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Int64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt8Feature) -> UInt64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt16Feature) -> UInt64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt32Feature) -> UInt64Feature:
        ...

    @overload
    def __floordiv__(self, other: UInt64Feature) -> UInt64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float32Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: Float64Feature) -> Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: int) -> UInt64Feature | Int64Feature:
        ...

    @overload
    def __floordiv__(self, other: float) -> Int64Feature:
        ...

    def __floordiv__(self, other: Any) -> Feature:
        """Performs floor division operation ('//').

        Returns:
            Feature: A new feature representing the integer quotient of the original features.
        """
        return self.execute_method("__floordiv__", other)

    @overload
    def __rfloordiv__(self, other: int) -> UInt64Feature | Int64Feature:
        ...

    @overload
    def __rfloordiv__(self, other: float) -> Int64Feature:
        ...

    def __rfloordiv__(self, other: Any) -> Feature:
        """Performs true right division operation with a constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the constant and the original
            feature.
        """
        return type(self).get_method("__floordiv__")(other, self)


class Float32Feature(
    PrimitiveFeature[dtypes.Float32Type], ComparableFeatureMixin, StatisticalFeatureMixin
):
    """A primitive feature representing a 32-bit floating-point number."""

    _expected_dtype: Final[dtypes.Type] = dtypes.Float32Type

    def mean(self) -> Float32Feature:
        """Calculates the mean (average) of the feature's values.

        This method computes the average of the values associated with the
        feature.

        Returns:
            Float32Feature: A new feature representing the mean of the original feature's values.
        """
        return super(Float32Feature, self).mean()

    def __abs__(self) -> Float32Feature:
        """Performs the absolute value operation ('abs').

        Returns:
            Float32Feature: A new feature representing the absolute value of the original feature.
        """
        return self.execute_method("__abs__")

    def __neg__(self) -> Float32Feature:
        """Performs the negation operation ('-').

        Returns:
            Float32Feature: A new feature representing the negated value of the original feature.
        """
        return self.execute_method("__neg__")

    @overload
    def __add__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __add__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __add__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
            | int
            | float
        ),
    ) -> Float32Feature:
        ...

    def __add__(self, other: Any) -> Feature:
        """Performs addition operation ('+').

        Returns:
            Feature: A new feature representing the sum of the original features.
        """
        return self.execute_method("__add__", other)

    def __radd__(self, other: int | float) -> Float32Feature:
        """Performs right addition operation with constant ('+').

        Returns:
            Feature: A new feature representing the sum of the original features.
        """
        return self.execute_method("__add__", other)

    @overload
    def __sub__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __sub__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __sub__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
            | int
            | float
        ),
    ) -> Float32Feature:
        ...

    def __sub__(self, other: Any) -> Feature:
        """Performs subtraction operation ('-').

        Returns:
            Feature: A new feature representing the difference of the original features.
        """
        return self.execute_method("__sub__", other)

    def __rsub__(self, other: int | float) -> Float32Feature:
        """Performs right subtraction operation with constant ('-').

        Returns:
            Float32Feature: A new feature representing the difference of the original features.
        """
        return type(self).get_method("__sub__")(other, self)

    @overload
    def __mul__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __mul__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __mul__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
            | int
            | float
        ),
    ) -> Float32Feature:
        ...

    def __mul__(self, other: Any) -> Feature:
        """Performs multiplication operation ('*').

        Returns:
            Feature: A new feature representing the product of the original features.
        """
        return self.execute_method("__mul__", other)

    def __rmul__(self, other: int | float) -> Float32Feature:
        """Performs right multiplication operation with constant ('*').

        Returns:
            Feature: A new feature representing the product of the original features.
        """
        return self.execute_method("__mul__", other)

    @overload
    def __truediv__(self, other: Float32Feature) -> Float32Feature:
        ...

    @overload
    def __truediv__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __truediv__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
            | int
            | float
        ),
    ) -> Float32Feature:
        ...

    def __truediv__(self, other: Any) -> Feature:
        """Performs true division operation ('/').

        Returns:
            Feature: A new feature representing the quotient of the original features.
        """
        return self.execute_method("__truediv__", other)

    def __rtruediv__(self, other: int | float) -> Float32Feature:
        """Performs true true right division operation with constant ('/').

        Returns:
            Float32Feature: A new feature representing the quotient of the original features.
        """
        return type(self).get_method("__truediv__")(other, self)

    def __floordiv__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
            | Float32Feature
            | Float64Feature
            | int
            | float
        ),
    ) -> Int64Feature:
        """Performs floor division operation ('//').

        Returns:
            Int64Feature: A new feature representing the integer quotient of the original features.
        """
        return self.execute_method("__floordiv__", other)

    def __rfloordiv__(self, other: int | float) -> Int64Feature:
        """Performs true right floor division operation with constant ('//').

        Returns:
            Feature: A new feature representing the integer quotient of the original features.
        """
        return type(self).get_method("__floordiv__")(other, self)


class Float64Feature(
    PrimitiveFeature[dtypes.Float64Type], ComparableFeatureMixin, StatisticalFeatureMixin
):
    """A primitive feature representing a 64-bit floating-point number."""

    _expected_dtype: Final[dtypes.Type] = dtypes.Float64Type

    def __abs__(self) -> Float64Feature:
        """Performs the absolute value operation ('abs').

        Returns:
            Float64Feature: A new feature representing the absolute value of the original feature.
        """
        return self.execute_method("__abs__")

    def __neg__(self) -> Float64Feature:
        """Performs the negation operation ('-').

        Returns:
            Float64Feature: A new feature representing the negated value of the original feature.
        """
        return self.execute_method("__neg__")

    @overload
    def __add__(self, other: Float32Feature) -> Float64Feature:
        ...

    @overload
    def __add__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __add__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
            | int
            | float
        ),
    ) -> Float64Feature:
        ...

    def __add__(self, other: Any) -> Feature:
        """Performs addition operation ('+').

        Returns:
            Feature: A new feature representing the sum of the original features.
        """
        return self.execute_method("__add__", other)

    def __radd__(self, other: int | float) -> Float64Feature:
        """Performs right addition operation with constant ('+').

        Returns:
            Feature: A new feature representing the sum of the original features.
        """
        return self.execute_method("__add__", other)

    @overload
    def __sub__(self, other: Float32Feature) -> Float64Feature:
        ...

    @overload
    def __sub__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __sub__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
            | int
            | float
        ),
    ) -> Float64Feature:
        ...

    def __sub__(self, other: Any) -> Feature:
        """Performs subtraction operation ('-').

        Returns:
            Feature: A new feature representing the difference of the original features.
        """
        return self.execute_method("__sub__", other)

    def __rsub__(self, other: int | float) -> Float64Feature:
        """Performs right subtraction operation with constant ('-').

        Returns:
            Float64Feature: A new feature representing the difference of the original features.
        """
        return type(self).get_method("__sub__")(other, self)

    @overload
    def __mul__(self, other: Float32Feature) -> Float64Feature:
        ...

    @overload
    def __mul__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __mul__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
            | int
            | float
        ),
    ) -> Float64Feature:
        ...

    def __mul__(self, other: Any) -> Feature:
        """Performs multiplication operation ('*').

        Returns:
            Feature: A new feature representing the product of the original features.
        """
        return self.execute_method("__mul__", other)

    def __rmul__(self, other: int | float) -> Float64Feature:
        """Performs right multiplication operation with constant ('*').

        Returns:
            Float64Feature: A new feature representing the product of the original features.
        """
        return self.execute_method("__mul__", other)

    @overload
    def __truediv__(self, other: Float32Feature) -> Float64Feature:
        ...

    @overload
    def __truediv__(self, other: Float64Feature) -> Float64Feature:
        ...

    @overload
    def __truediv__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
            | int
            | float
        ),
    ) -> Float64Feature:
        ...

    def __truediv__(self, other: Any) -> Feature:
        """Performs true division operation ('/').

        Returns:
            Feature: A new feature representing the quotient of the original features.
        """
        return self.execute_method("__truediv__", other)

    def __rtruediv__(self, other: int | float) -> Float64Feature:
        """Performs true true right division operation with constant ('/').

        Returns:
            Feature: A new feature representing the quotient of the original features.
        """
        return type(self).get_method("__truediv__")(other, self)

    def __floordiv__(
        self,
        other: (
            Int8Feature
            | Int16Feature
            | Int32Feature
            | Int64Feature
            | UInt8Feature
            | UInt16Feature
            | UInt32Feature
            | UInt64Feature
            | Float32Feature
            | Float64Feature
            | int
            | float
        ),
    ) -> Int64Feature:
        """Performs floor division operation ('//').

        Returns:
            Int64Feature: A new feature representing the integer quotient of the original features.
        """
        return self.execute_method("__floordiv__", other)

    def __rfloordiv__(self, other: int | float) -> Int64Feature:
        """Performs true right floor division operation with constant ('//').

        Returns:
            Feature: A new feature representing the integer quotient of the original features.
        """
        return type(self).get_method("__floordiv__")(other, self)


@dataclass(eq=True, frozen=False)
class ClassLabelFeature(Int64Feature):
    """A base class for defining strongly-typed categorical class labels.

    This class provides a mechanism for defining and validating class labels, typically used to
    represent categorical data in datasets. Each class label corresponds to an integer ID and is
    mapped to an :class:`Int64Feature` instance. Subclasses of :class:`ClassLabelFeature` are
    intended to define the set of class labels in a similar way to an :class:`IntEnum`.

    Subclasses automatically inherit functionality for validation and schema generation, ensuring
    that the defined labels are consistent and complete. Missing label IDs are identified and
    handled during validation.

    Defining Class Labels
    ~~~~~~~~~~~~~~~~~~~~~

    Subclassing :class:`ClassLabelFeature` allows users to define class labels as attributes,
    similar to defining members in an `Enum`:

    Example:
    .. code-block:: python

        class Labels(ClassLabelFeature):
            FIRST = 0
            SECOND = 1

        # This defines a feature with two class labels:
        # - Label `FIRST` corresponds to ID 0.
        # - Label `SECOND` corresponds to ID 1.
    """

    def __post_init__(self) -> None:
        """Validates the class label data type and ensures label definitions are consistent."""
        assert isinstance(self.dtype, dtypes.ClassLabelType)

        # make sure that the members that are specified in the class label enum
        # are valid in the class label data type
        for member in self._build_class_label_enum():
            if (member.value >= len(self.dtype)) or (self.dtype.names[member.value] != member.name):
                raise RuntimeError(
                    f"Label mismatch detected in {self.__class__.__qualname__}: "
                    f"The label '{member.name}' (ID {member.value}) is invalid. "
                    + (
                        "The ID exceeds the number of labels defined in the data type."
                        if member.value >= len(self.dtype)
                        else f"Expected label name: {self.dtype.names[member.value]}"
                    )
                )

    @classmethod
    def _build_class_label_enum(cls) -> IntEnum:
        """Constructs an enum of class labels defined in this class.

        Returns:
            IntEnum: An enumeration of the class labels.
        """
        # TODO: this only captures the values defined in this class
        #       but not those inherited from base types
        values = {
            k: v for k, v in vars(cls).items() if isinstance(v, int) and not k.startswith("__")
        }
        return IntEnum("ClassLabelEnum", values)

    @classmethod
    def _build_class_label_dtype(cls) -> dtypes.ClassLabelType:
        """Constructs the class label data type based on the defined labels.

        Returns:
            types.ClassLabelType: The constructed class label data type.
        """
        names = []
        enum = cls._build_class_label_enum()
        for i in range(0, max(enum, default=-1) + 1):
            if i not in set(enum):
                warnings.warn(
                    f"Detected missing label ID {i} in {cls.__qualname__}, filling with 'UNDEF'."
                )
            names.append(enum(i).name if i in set(enum) else "UNDEF")
        return dtypes.ClassLabelType(names=tuple(names))

    @classmethod
    def from_names(cls, names: list[str]) -> type[ClassLabelFeature]:
        """Dynamically creates class label type from a list of class label names.

        This method dynamically defines a custom :class:`ClassLabelFeature` subclass by providing
        a list of class label names. Each label is automatically assigned a unique integer ID
        starting from 0, based on its position in the list.

        This is particularly useful when the class labels are dynamically generated or need
        to be defined programmatically, instead of being hardcoded as class-level attributes.

        Args:
            names (list[str]): A list of class label names. Each name in the list represents
                a unique label, and the index of the name determines its corresponding integer ID.

        Returns:
            type[ClassLabelFeature]: A dynamically created subclass of :class:`ClassLabelFeature`,
            where each label name in the input list is assigned as a class-level attribute with its
            corresponding integer ID as the value.

        Example:
        --------
        .. code-block:: python

            from my_module import ClassLabelFeature

            # Define class labels programmatically
            label_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
            CustomLabels = ClassLabelFeature.from_names(label_names)

            # Access the dynamically created class labels
            print(CustomLabels.NEGATIVE)  # Output: 0
            print(CustomLabels.NEUTRAL)  # Output: 1
            print(CustomLabels.POSITIVE)  # Output: 2
        """
        return type(
            f"ClassLabelFeature[{','.join(names)}]",
            (cls,),
            {name: i for i, name in enumerate(names)},
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Builds the Pydantic core schema for the class label feature.

        Args:
            source_type (Any): The type being validated.
            handler (pydantic.GetCoreSchemaHandler): A handler function used to the core schema.

        Returns:
            core_schema.CoreSchema: The Pydantic core schema for the mapping feature,
                which is used to validate instances of the mapping.

        Raises:
            PydanticCustomError: If the data type of the instance does not match
                the expected :code:`dtype`.
        """

        def validator_fn(inst: Feature | BaseReference, validator: Callable[[Any], Any]) -> Feature:
            # TODO: only convert in strict mode, see mapping feature
            if isinstance(inst, ClassLabelFeature) and (cls != ClassLabelFeature):
                # convert the class label feature to the specific
                # class label feature instance
                inst = cls(inst.ref)

            elif isinstance(inst, BaseReference):
                if inst.get_dtype() is None:
                    # fallback to the data type inferred from the specific subclass
                    # in case the reference does not specify a dtype
                    assert isinstance(inst, ForwardReference)
                    inst = ForwardReference(cls._build_class_label_dtype())

                # create a class label feature instance from the reference
                inst = cls(inst)

                if len(inst.dtype) == 0:
                    raise PydanticCustomError(
                        "Invalid Class Labels",
                        (
                            "No class labels defined in '{class_name}'. Ensure that the class "
                            "defines at least one label as a class-level attribute."
                        ),
                        {"class_name": cls.__qualname__},
                    )

            # run base validator
            return validator(inst)

        return core_schema.no_info_wrap_validator_function(
            validator_fn, schema=core_schema.is_instance_schema(cls)
        )


T = TypeVar("T")


@dataclass(eq=True, frozen=False)
class SequenceFeature(typing.Sequence[T], Feature[dtypes.SequenceType]):
    """A feature representing a sequence of items.

    The :class:`SequenceFeature` class models a feature where the data type is a sequence,
    supporting indexing, slicing, and length operations while maintaining type
    safety and feature reference consistency.
    """

    def __post_init__(self) -> None:
        """Validates that the data type is a valid sequence type."""
        assert isinstance(self.dtype, dtypes.SequenceType)

    @typing.overload
    def __getitem__(self, index: int) -> T:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> SequenceFeature[T]:
        ...

    def __getitem__(self, index: int | slice) -> T | SequenceFeature[T]:
        """Implements indexing or slicing for the sequence.

        If an integer index is provided, returns the corresponding element.
        If a slice is provided, returns a new :class:`SequenceFeature` feature for
        the sliced range.

        Args:
            index (int | slice): The index or range of indices to retrieve.

        Returns:
            T | SequenceFeature[T]: The element or the sliced sequence feature.
        """
        return self.execute_method("__getitem__", index)

    def __len__(self) -> int:
        """Raises :class:`EnvironmentError` to avoid confusion with the :code:`length` method.

        This method is intentionally not implemented to ensure that users explicitly
        use the :code:`length` method for determining the length of the sequence. The
        :code:`length` method supports dynamic resolution during execution.
        """
        raise EnvironmentError("Use '.length' instead of 'len()'.")

    def length(self) -> int | Int32Feature:
        """Returns the length of the sequence.

        Returns:
            int | Int32Feature: The number of elements in the sequence. An integer is returned if
            the length is fixed. Otherwise a feature is returned which resolves to the sequence
            length during execution.
        """
        return (
            len(self.dtype)
            if len(self.dtype) != dtypes.UNDEFINED_SEQUENCE_LENGTH
            else self.execute_method("length")
        )

    def min(self) -> T:
        """Returns the minimum value in the sequence.

        Returns:
            T: A feature representing the minimum value in the sequence.
        """
        return self.execute_method("min")

    def max(self) -> T:
        """Returns the maximum value in the sequence.

        Returns:
            T: A feature representing the maximum value in the sequence.
        """
        return self.execute_method("max")

    def sum(self) -> T:
        """Returns the sum of the sequence.

        Returns:
            T: A feature representing the sum of the sequence.
        """
        return self.execute_method("sum")

    U = TypeVar("U")

    def foreach(self, fn: Callable[[T], U]) -> SequenceFeature[U]:
        """Apply a transformation function to each element in the sequence.

        This method applies the provided transformation function to each element
        in the sequence.

        Args:
            fn (Callable[[T], U]): A function to apply to each element of the sequence.

        Returns:
            SequenceFeature[U]: A new :class:`SequenceFeature` instance containing the
            transformed sequence, with the same structural layout as the original.
        """
        values, index = self.unpack(return_index=True)
        return SequenceFeature.get_method("pack")(fn(values), index, values.ref)

    @overload
    def unpack(self) -> T:
        ...

    @overload
    def unpack(self, return_index: typing.Literal[False]) -> T:
        ...

    @overload
    def unpack(self, return_index: typing.Literal[True]) -> tuple[T, Int32Feature]:
        ...

    def unpack(self, return_index: bool = False) -> T | tuple[T, Int32Feature]:
        """Unpack the sequence into its values.

        This method retrieves the values of the sequence, and if :code:`return_index` is set to
        :code:`True`, it also includes the index mapping that relates the values to their
        original structure.

        Args:
            return_index (bool, optional): If :code:`True`, the method returns both the sequence
                values and their indices. Defaults to :code:`False`.

        Returns:
            T | tuple[T, Int32Feature]: The unpacked sequence values and optionally the trace
                indices if :code:`return_index` is set to :code:`True`.
        """
        if return_index:
            values_with_index = self.execute_method("unpack_with_index")
            return values_with_index["value"], values_with_index["index"]
        else:
            return self.execute_method("unpack")

    @classmethod
    def __init_subclass__(cls) -> None:
        """Prevents subclassing of the :class:`SequenceFeature` class."""
        raise EnvironmentError("Cannot inherit from type")

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Builds the pydantic core schema for the :class:`SequenceFeature` class.

        This method creates a schema that performs two key operations:

        1. If the input is a :class:`BaseReference` object, this method constructs a
           corresponding :class:`SequenceFeature` feature.

        2. Ensures that the value type of the :class:`SequenceFeature` instance matches
           the expected value type.

        The expected value type is inferred from the generic type annotation. For example,
        if the annotation is :code:`Sequence[Int]`, the value type is inferred as :code`Int`.
        This inferred type is used both to validate elements of the sequence and to construct
        a sequence feature from a reference.

        Args:
            source_type (Any): The source type being validated, including any generic
                parameters that specify the expected value type.
            handler (pydantic.GetCoreSchemaHandler): A handler function for generating the
                core schema.

        Returns:
            core_schema.CoreSchema: A schema that validates `SequenceFeature` instances
            and enforces type constraints for sequence features.
        """
        adapter: pydantic.TypeAdapter = None

        if isinstance(source_type, GenericAlias):
            # get the expected item type from the generic annotation
            assert typing.get_origin(source_type) is cls
            (expected_item_type,) = typing.get_args(source_type)
            # create the type adapter to validate the item type
            config = (
                None
                if _type_has_config(expected_item_type)
                else pydantic.ConfigDict(arbitrary_types_allowed=True)
            )
            adapter = pydantic.TypeAdapter(expected_item_type, config=config)

        def validator_fn(inst, validator, info):
            if isinstance(inst, BaseReference):
                if inst.get_dtype() is not None:
                    return SequenceFeature(inst)

                if adapter is None:
                    # no value type specified
                    raise RuntimeError(
                        "Cannot validate a Sequence feature from a Reference without specifying "
                        "the value type. Ensure that the generic type alias (e.g., Sequence[Int]) "
                        "is provided to define the expected value type."
                    )

                assert isinstance(inst, ForwardReference)
                # create instance from reference
                value_feature: Feature = adapter.validate_python(inst, context=info.context)
                dtype = dtypes.SequenceType(value_feature.dtype)
                inst = build_feature_from_reference(ForwardReference(dtype))

            # run the core validator checking that the instance
            # is a valid sequence feature
            validator(inst)

            if adapter is not None:
                # create a copy of the instance but with length 1
                # and validate the value feature at position 0 as a representative
                # of all the sequence values
                value_feature = build_feature_from_reference(
                    ForwardReference(inst.dtype.value_type)
                )
                adapter.validate_python(value_feature, context=info.context, strict=True)

            return inst

        return core_schema.with_info_wrap_validator_function(
            validator_fn, schema=core_schema.is_instance_schema(SequenceFeature)
        )


@dataclass(eq=True, frozen=False)
class _MappingFeature(typing.Mapping, Feature[dtypes.MappingType]):
    """A base class for defining strongly-typed mappings.

    Represents a strongly-typed mapping feature, which can be used to define fields
    of a mapping (similar to :class:`TypedDict`) with specified key-value pairs where
    the values are instances of :class:`_Feature` types.

    This class is designed to be subclassed, where the fields of the subclass are
    specified in a manner similar to :class:`TypedDict`. Each field corresponds to a
    key in the mapping, and the values of these fields are expected to be features with
    a specific data type. Subclasses of :class:`_MappingFeature` automatically inherit
    the validation mechanism that ensures the keys and values adhere to the expected
    data types.

    Example:
    .. code-block:: python

        class MyMappingFeature(_MappingFeature):
            key1: _String
            key2: _Int32

        # This subclass defines a mapping with `key1` as a string feature
        # and `key2` as an integer feature.
    """

    def __post_init__(self) -> None:
        """Post initialization validation.

        Ensures that the :code:`dtype` attribute is of type :class:`types.MappingType`
        and validates that the subclass is correctly configured with the expected keys.

        This method performs two main tasks:

        1. Verifies that the :code:`dtype` is of the correct type, i.e.,
           :class:`types.MappingType`, to confirm that the instance represents
           a mapping feature.
        2. Checks that the fields defined in the subclass match the keys specified in
           the :code:`dtype`, ensuring that no invalid or missing keys are present. If
           there are any issues with the keys (invalid or missing), a :class:`KeyError`
           is raised.

        The method ensures that the subclass follows the expected structure, where the fields
        specified in the subclass are validated against the actual data in :code:`dtype`.

        Raises:
            KeyError: If there are any invalid or missing keys in the
                      mapping feature.
        """
        assert isinstance(self.dtype, dtypes.MappingType)

        # base mapping type allows arbitrary keys
        if type(self) is _MappingFeature:
            return

        # get the set of valid keys
        valid_keys = set(typing.get_type_hints(type(self)).keys()) - set(
            typing.get_type_hints(_MappingFeature).keys()
        )
        # get the specified keys from the factory entries
        set_keys = set(self.keys())
        # compute the invalid and missing keys
        invalid_keys = set_keys - valid_keys
        missing_keys = valid_keys - set_keys

        if len(invalid_keys) > 0:
            raise KeyError(f"Invalid Keys: {invalid_keys}", invalid_keys)

        if len(missing_keys) > 0:
            raise KeyError(f"Missing Keys: {missing_keys}", missing_keys)

    def __len__(self) -> int:
        """Returns the number of elements in the mapping."""
        return len(self.dtype)

    def __iter__(self) -> typing.Iterable[str]:
        """Returns an iterator over the keys in the mapping."""
        return iter(self.dtype)

    def __getitem__(self, key: str) -> Feature:
        """Retrieves the feature corresponding to the specified key.

        Args:
            key (str): The key to retrieve the feature for.

        Returns:
            _Feature: The feature associated with the key.
        """
        return self.execute_method("__getitem__", str(key))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Builds the Pydantic core schema for the :class:`_MappingFeature` class.

        This method generates a Pydantic model that mirrors the structure of the mapping feature.
        The generated model is used to validate the fields of the mapping, ensuring they adhere
        to the expected types and structure.

        The validation process includes two main tasks:

        1. If the input is a :class:`BaseReference` object, the method constructs a corresponding
        :class:`_MappingFeature` instance.
        2. It ensures that the fields of the :class:`_MappingFeature` instance match the expected
        fields and data types as defined by the Pydantic model.

        Args:
            source_type (Any): The type being validated, which may include generic parameters
                specifying the expected field types.
            handler (pydantic.GetCoreSchemaHandler): A handler function used to generate the
                core schema.

        Returns:
            core_schema.CoreSchema: The Pydantic core schema for the mapping feature,
                which is used to validate instances of the mapping.

        Raises:
            RuntimeError: If a :class:`BaseReference` is provided, but no corresponding
                Pydantic model can be created for validation.
        """
        ignore_keys = set(typing.get_type_hints(_MappingFeature).keys())

        def build_validator_model(cls) -> pydantic.BaseModel:
            if not (
                (
                    isinstance(cls, type)
                    and issubclass(cls, _MappingFeature)
                    and (cls is not _MappingFeature)
                )
                or (
                    isinstance(cls, GenericAlias)
                    and issubclass(typing.get_origin(cls), _MappingFeature)
                )
            ):
                # trivial case: the class is not a subclass
                # of mapping or the base class itself
                return cls

            # get the annotations of only this type
            annotations = typing.get_type_hints(cls, include_extras=True)
            annotations = {
                k: (h, pydantic.Field()) for k, h in annotations.items() if k not in ignore_keys
            }
            # prepare the bsae classes
            bases = get_original_bases(cls)
            bases = map(build_validator_model, bases)
            bases = tuple(b for b in bases if b is not _MappingFeature)
            # add the pydantic base model as a base class
            if not any(
                (isinstance(base, type) and issubclass(base, pydantic.BaseModel)) for base in bases
            ):
                bases = (BaseModelWithArbitraryTypesAllowed,) + bases

            # create the validator model
            model = pydantic.create_model(
                f"MappingValidatorFor{cls.__name__}",
                **annotations,
                __base__=bases,
            )
            # apply the type variable values if the model is generic
            if isinstance(cls, GenericAlias):
                model = model.__class_getitem__(typing.get_args(cls))

            return model

        # build the validation model in case in case the class is not the base mapping
        # class itself, otherwise no field validation is done
        model = build_validator_model(source_type) if cls is not _MappingFeature else None
        model = model if model is not _MappingFeature else None

        def validator_fn(inst, validator, info):
            if isinstance(inst, BaseReference):
                if inst.get_dtype() is not None:
                    return _MappingFeature(inst)

                if model is None:
                    raise RuntimeError(
                        "Cannot validate a Mapping feature from a Reference without "
                        "specifying fields."
                    )

                assert isinstance(inst, ForwardReference)
                # infer member types from annotations using the validation model
                members = {key: ForwardReference() for key in model.model_fields.keys()}
                members = {key: field.dtype for key, field in model.model_validate(members)}
                # create the mapping instance from the member types
                dtype = dtypes.MappingType.construct(members)
                inst = build_feature_from_reference(ForwardReference(dtype))

            # run the core validator checking that the instance
            # is a valid mapping
            validator(inst)

            if model is not None:
                # validate the field types
                fields = {
                    key: build_feature_from_reference(ForwardReference(inst.dtype[key]))
                    for key in inst.keys()
                }
                model.model_validate(fields, context=info.context, strict=True)

            strict = (info.context or {}).get("strict", False)
            # convert the instance to the actual class type in case of
            # strict validation
            return inst if isinstance(inst, cls) or not strict else cls(inst.ref)

        return core_schema.with_info_wrap_validator_function(
            validator_fn, schema=core_schema.is_instance_schema(_MappingFeature)
        )


if typing.TYPE_CHECKING:  # pragma: not covered
    # If we are type checking (during static analysis), define the `MappingFeature` as a
    # TypedDict. This allows us to specify a dictionary-like structure with the correct
    # types for static type checking.

    class MappingFeature(typing.TypedDict, _MappingFeature):
        """A base class for defining strongly-typed mappings.

        Represents a strongly-typed mapping feature, which can be used to define fields
        of a mapping (similar to :class:`TypedDict`) with specified key-value pairs where
        the values are instances of :class:`_Feature` types.

        This class is designed to be subclassed, where the fields of the subclass are
        specified in a manner similar to :class:`TypedDict`. Each field corresponds to a
        key in the mapping, and the values of these fields are expected to be features with
        a specific data type. Subclasses of :class:`_MappingFeature` automatically inherit
        the validation mechanism that ensures the keys and values adhere to the expected
        data types.

        Example:
        .. code-block:: python

            class MyMappingFeature(MappingFeature):
                key1: _String
                key2: _Int32

            # This subclass defines a mapping with `key1` as a string feature
            # and `key2` as an integer feature.
        """

else:
    # At runtime, we use a type alias to assign `_MappingFeature` as the concrete implementation
    # of `MappingFeature`. This ensures that during execution, `MappingFeature` is treated as
    # `_MappingFeature`, which is the actual class.

    MappingFeature: typing.TypeAlias = _MappingFeature

PRIMITIVE_FEATURE_MAPPING = {
    dtypes.BoolType: BoolFeature,
    dtypes.StringType: StringFeature,
    dtypes.UInt8Type: UInt8Feature,
    dtypes.UInt16Type: UInt16Feature,
    dtypes.UInt32Type: UInt32Feature,
    dtypes.UInt64Type: UInt64Feature,
    dtypes.Int8Type: Int8Feature,
    dtypes.Int16Type: Int16Feature,
    dtypes.Int32Type: Int32Feature,
    dtypes.Int64Type: Int64Feature,
    dtypes.Float32Type: Float32Feature,
    dtypes.Float64Type: Float64Feature,
}


def build_feature_from_reference(
    ref: BaseReference, fallback_dtype: None | dtypes.Type = None
) -> Feature:
    """Build a feature from a given reference.

    This function takes a reference and a data type, and constructs the appropriate
    feature based on the type of the data.

    Args:
        ref (BaseReference): The reference to the feature being created.
        fallback_dtype (None | dtypes.Type): The fallback dtype used in case the dtype
            cannot be inferred from the reference.

    Returns:
        Feature: The corresponding feature based on the type of :code:`dtype`.

    Raises:
        TypeError: If the :code:`dtype` is not recognized.
    """
    # infer the dtype of the feature from the reference or the fallback
    ref = ref if ref.get_dtype() is not None else ForwardReference(fallback_dtype)
    dtype = ref.get_dtype()

    if dtype is None:
        raise RuntimeError()  # TODO: error message, dtype not defined

    if isinstance(dtype, dtypes.ClassLabelType):
        return ClassLabelFeature(ref)

    elif isinstance(dtype, dtypes.PrimitiveType):
        return PRIMITIVE_FEATURE_MAPPING[dtype](ref)

    elif isinstance(dtype, dtypes.SequenceType):
        return SequenceFeature(ref)

    elif isinstance(dtype, dtypes.MappingType):
        return MappingFeature(ref)

    raise TypeError(f"Unsupported data type, got {dtype}.")


def build_feature_from_annotation(
    annotation: Any,
    typevar_mapping: dict[TypeVar, dtypes.Type] = {},
    context: dict[str, Any] = {},
) -> Feature:
    """Build a feature from a given annotation using a forward reference.

    This function inspects the provided annotation and resolves any type parameters
    or type variables, then constructs a feature accordingly. If the annotation
    includes type parameters, it creates a generic validation model and resolves
    the correct feature types. This function supports features with generic annotations
    and resolves them to concrete feature types.

    The underlying reference is a :class:`ForwardReference` instance with a data type
    corresponding to the resolved feature type.

    Args:
        annotation (Any): The annotation that describes the feature's type, which can
            include type parameters or type variables.
        typevar_mapping (dict[TypeVar, types.Type]): A mapping that associates
            type variables with their corresponding types. Defaults to an empty dictionary.
        context (dict[str, Any]): A context dictionary that can provide additional
            information to the validation process. Defaults to an empty dictionary.

    Returns:
        Feature: The feature built from the annotation and reference. This feature matches
            the structure defined by the annotation and resolves any type parameters.

    """

    def maybe_build_feature_from_reference(
        inst: Feature | BaseReference, dtype: dtypes.Type
    ) -> Feature:
        return (
            inst
            if isinstance(inst, Feature)
            else build_feature_from_reference(inst, fallback_dtype=dtype)
        )

    # get the return annotation
    has_parameters = hasattr(annotation, "__parameters__") and len(annotation.__parameters__) > 0

    base = (BaseModelWithArbitraryTypesAllowed,)

    if has_parameters:
        # add generic base in case the return annotation has any parameters
        base += (typing.Generic[annotation.__parameters__],)  # type: ignore

    elif isinstance(annotation, TypeVar):
        # add generic base in case the return annotation is just a typevar
        base += (typing.Generic[annotation],)  # type: ignore

    # create the validation model
    builder = pydantic.create_model(
        f"FeatureBuilder({annotation})", field=(annotation, pydantic.Field()), __base__=base
    )

    if hasattr(builder, "__parameters__"):
        # lookup typevars in return annotation
        dtypes = [typevar_mapping[t] for t in builder.__parameters__]
        # create a feature annotation for each typevar
        # which resolves to the corresponding feature type
        features = [
            typing.Annotated[
                Feature,
                pydantic.BeforeValidator(partial(maybe_build_feature_from_reference, dtype=dtype)),
            ]
            for dtype in dtypes
        ]
        # apply the feature annotations to the return model
        builder = builder.__class_getitem__(*features)

    # build the output feature
    feature = builder.model_validate({"field": ForwardReference()}, context=context).field
    assert isinstance(feature, Feature)

    return feature
