"""This module defines a collection of data processors that implement common numeric operations.

Each processor is designed to handle a specific numeric computation, such as addition, subtraction,
negation, or computing absolute values. These processors are intended for use in data processing
pipelines, enabling efficient and batched execution using Apache Arrow as the backend.

These processors are registered as methods on the respective numeric feature classes, such as
:class:`IntFeature` and :class:`FloatFeature`, allowing them to be applied directly to numeric
features.
"""
from typing import Annotated, Callable, TypeVar

import pyarrow.compute as pc

from ..features.features import (
    Feature,
    Float16Feature,
    Float32Feature,
    Float64Feature,
    Int8Feature,
    Int16Feature,
    Int32Feature,
    Int64Feature,
    UInt8Feature,
    UInt16Feature,
    UInt32Feature,
    UInt64Feature,
)
from ..features.validators import TypeResolver
from ..nodes.base import RunContext, process_mode
from ..nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from ..typing import Float, Int, UInt

ScalarType = TypeVar("ScalarType", bound=Float | Int | UInt)

Fn = TypeVar("Fn", bound=Callable)


def register_all(name: str, types: type[Feature]) -> Callable[[Fn], Fn]:
    """Helper function to register a method to multiple feature types.

    Args:
        name (str): The name of the method to register.
        types (type[Feature]): A collection of feature types for which the method will be
            registered.

    Returns:
        Callable[[Fn], Fn]: A decorator that wraps the function, registering it for the
            specified feature types.
    """

    def wrapper(fn):
        for feature_type in types:
            feature_type.register_method(name)(fn)
        return fn

    return wrapper


class AbsConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`Abs` processor."""


class Abs(BaseDataProcessor[AbsConfig]):
    """Data Processor for computing absolute values."""

    T = TypeVar("T", bound=Int | Float | UInt)

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, x: T) -> T:
        """Computes the absolute value of the input.

        Args:
            ctx (RunContext): The execution context for the processor.
            x (T): The numeric input data.

        Returns:
            T: The absolute value of the input data.
        """
        return pc.abs(x)


class NegateConfig(BaseDataProcessorConfig):
    """Configuration for the Negate processor."""


class Negate(BaseDataProcessor[NegateConfig]):
    """Data Processor for negating numeric values."""

    T = TypeVar("T", bound=Int | UInt | Float)

    @process_mode(batched=True, backend="arrow")
    def process(
        self, ctx: RunContext, x: T
    ) -> Annotated[
        Int | Float,
        TypeResolver(
            lambda _, inputs, session: (
                Int16Feature
                if isinstance(inputs["x"], UInt8Feature)
                else Int32Feature
                if isinstance(inputs["x"], UInt16Feature)
                else Int64Feature
                if isinstance(inputs["x"], UInt32Feature)
                else Int64Feature
                if isinstance(inputs["x"], UInt64Feature)
                else Negate.T
            )
        ),
    ]:
        """Computes the negation of the input.

        Args:
            ctx (RunContext): The execution context for the processor.
            x (T): The numeric input data.

        Returns:
            T: The negated value of the input data.
        """
        return pc.negate(pc.cast(x, ctx.output_type.arrow_type))


class AddConfig(BaseDataProcessorConfig):
    """Configuration for the Add processor."""


class Add(BaseDataProcessor[AddConfig]):
    """Data Processor implementing addition.

    This processor takes two numeric inputs, :code:`x` and :code:`y`, and
    computes their sum.
    """

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, x: ScalarType, y: ScalarType) -> ScalarType:
        """Perform addition of two numeric inputs.

        Args:
            ctx (RunContext): The execution context for the processor.
            x (Numeric): The first numeric operand.
            y (Numeric): The second numeric operand.

        Returns:
            Numeric: The result of adding :code:`x` and :code:`y`.
        """
        return pc.add(x, y)


class SubtractConfig(BaseDataProcessorConfig):
    """Configuration for the Subtract processor."""


class Subtract(BaseDataProcessor[SubtractConfig]):
    """Data Processor implementing subtraction.

    This processor takes two numeric inputs, :code:`x` and :code:`y`, and
    computes their difference (:code:`x - y`).
    """

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, x: ScalarType, y: ScalarType) -> ScalarType:
        """Perform subtraction of two numeric inputs.

        Args:
            ctx (RunContext): The execution context for the processor.
            x (ScalarType): The first numeric operand.
            y (ScalarType): The second numeric operand.

        Returns:
            ScalarType: The result of subtracting :code:`y` from :code:`x`.
        """
        return pc.subtract(x, y)


class MultiplyConfig(BaseDataProcessorConfig):
    """Configuration for the Multiply processor."""


class Multiply(BaseDataProcessor[MultiplyConfig]):
    """Data Processor implementing multiplication.

    This processor takes two numeric inputs, :code:`x` and :code:`y`, and
    computes their product (:code:`x * y`).
    """

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, x: ScalarType, y: ScalarType) -> ScalarType:
        """Perform multiplication of two numeric inputs.

        Args:
            ctx (RunContext): The execution context for the processor.
            x (ScalarType): The first numeric operand.
            y (ScalarType): The second numeric operand.

        Returns:
            ScalarType: The product of :code:`x` and :code:`y`.
        """
        return pc.multiply(x, y)


class TrueDivConfig(BaseDataProcessorConfig):
    """Configuration for the true division processor."""


class TrueDiv(BaseDataProcessor[TrueDivConfig]):
    """Data Processor implementing true division.

    This processor takes two numeric inputs, :code:`x` and :code:`y`, and
    computes their quotient (:code:`x / y`).
    """

    @process_mode(batched=True, backend="arrow")
    def process(
        self, ctx: RunContext, x: ScalarType, y: ScalarType
    ) -> Annotated[
        Float,
        TypeResolver(
            lambda _, inputs, session: (
                ScalarType
                if (
                    isinstance(inputs["x"], (Float16Feature, Float32Feature, Float64Feature))
                    or isinstance(inputs["y"], (Float16Feature, Float32Feature, Float64Feature))
                )
                else Float64Feature
            )
        ),
    ]:
        """Perform division of two numeric inputs.

        Args:
            ctx (RunContext): The execution context for the processor.
            x (ScalarType): The numerator.
            y (ScalarType): The denominator.

        Returns:
            Float: The quotient of :code:`x` and :code:`y`.
        """
        return pc.divide(
            pc.cast(x, ctx.output_type.arrow_type), pc.cast(y, ctx.output_type.arrow_type)
        )


class FloorDivConfig(BaseDataProcessorConfig):
    """Configuration for the floor division processor."""


class FloorDiv(BaseDataProcessor[TrueDivConfig]):
    """Data Processor implementing floor division.

    This processor takes two numeric inputs, :code:`x` and :code:`y`, and
    computes their integer quotient (:code:`x // y`).
    """

    @process_mode(batched=True, backend="arrow")
    def process(
        self, ctx: RunContext, x: ScalarType, y: ScalarType
    ) -> Annotated[
        Int,
        TypeResolver(
            lambda _, inputs, session: (
                ScalarType
                if (
                    isinstance(inputs["x"], (Int8Feature, Int16Feature, Int32Feature, Int64Feature))
                    and isinstance(
                        inputs["y"], (Int8Feature, Int16Feature, Int32Feature, Int64Feature)
                    )
                )
                else Int64Feature
            )
        ),
    ]:
        """Perform floor division of two numeric inputs.

        Args:
            ctx (RunContext): The execution context for the processor.
            x (ScalarType): The numerator.
            y (ScalarType): The denominator.

        Returns:
            Float: The integer quotient of :code:`x` and :code:`y`.
        """
        return pc.cast(pc.floor(pc.divide(x, y)), ctx.output_type.arrow_type)


register_all(
    "__abs__",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        Float16Feature,
        Float32Feature,
        Float64Feature,
    ],
)(Abs().call)

# unsigned integers are always non-negative
# absolut value has no effect and can be reduced to no-op
register_all(
    "__abs__",
    [
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
    ],
)(lambda x: x)

register_all(
    "__neg__",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
        Float16Feature,
        Float32Feature,
        Float64Feature,
    ],
)(Negate().call)

register_all(
    "__add__",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
        Float16Feature,
        Float32Feature,
        Float64Feature,
    ],
)(Add().call)

register_all(
    "__sub__",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
        Float16Feature,
        Float32Feature,
        Float64Feature,
    ],
)(Subtract().call)

register_all(
    "__mul__",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
        Float16Feature,
        Float32Feature,
        Float64Feature,
    ],
)(Multiply().call)

register_all(
    "__truediv__",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
        Float16Feature,
        Float32Feature,
        Float64Feature,
    ],
)(TrueDiv().call)

register_all(
    "__floordiv__",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
        Float16Feature,
        Float32Feature,
        Float64Feature,
    ],
)(FloorDiv().call)
