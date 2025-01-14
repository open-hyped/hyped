"""This module defines a collection of data processors that implement common numeric operations.

Each processor is designed to handle a specific numeric computation, such as addition, subtraction,
negation, or computing absolute values. These processors are intended for use in data processing
pipelines, enabling efficient and batched execution using Apache Arrow as the backend.

These processors are registered as methods on the respective numeric feature classes, such as
:class:`IntFeature` and :class:`FloatFeature`, allowing them to be applied directly to numeric
features.
"""
from typing import Annotated, Any, Callable, TypeAlias, TypeVar

import pyarrow.compute as pc

from ..abc import AbstractDataFlowGraphBuilder
from ..features.dtypes import (
    DType,
    Float32Type,
    Float64Type,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    build_dtype_from_python_object,
)
from ..features.features import (
    Feature,
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
    build_feature_from_reference,
)
from ..features.reference import ConcreteReference
from ..features.validators import FeatureResolver
from ..nodes.aggregator import BaseDataAggregator, BaseDataAggregatorConfig
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


def add_constant(
    val: Any, candidate_dtype: DType, builder: AbstractDataFlowGraphBuilder
) -> Feature:
    """Adds a constant value to a data flow graph as a node and returns it as a :class:`Feature`.

    This function evaluates the provided constant value's data type. If the candidate
    data type matches the value (e.g., a float value with a float type or an integer
    value with an integer type), it uses the candidate type. Otherwise, it infers the
    data type from the value. The constant is then added to the graph as a node.

    Args:
        val (Any): The constant value to be added to the graph.
        candidate_dtype (DType): The candidate data type for the value, expected to
            be a type like :code:`Float32Type` or :code:`Int64Type`.
        builder (AbstractDataFlowGraphBuilder): The data flow graph builder to add the constant.

    Returns:
        Feature: A :class:`Feature` object representing the constant added to the graph.
    """
    if isinstance(val, float) and candidate_dtype in {Float32Type, Float64Type}:
        # candidate type is float and value is a float value
        # use the candidate data type to represent the value
        dtype = candidate_dtype

    elif (
        isinstance(val, int)
        and (val >= 0)
        and candidate_dtype in {UInt8Type, UInt16Type, UInt32Type, UInt64Type}
    ):
        # candidate type is an unsigned integer and value is a non-negative
        # integer value, use the candidate data type to represent the value
        dtype = candidate_dtype

    elif isinstance(val, int) and candidate_dtype in {Int8Type, Int16Type, Int32Type, Int64Type}:
        # candidate type is integer and value is an integer value
        # use the candidate data type to represent the value
        dtype = candidate_dtype

    else:
        # infer the data type from the value
        dtype = build_dtype_from_python_object(val)

    # add the constant node to the graph
    ref = builder.const_node(val, dtype)
    return build_feature_from_reference(ref)


def handle_constant_for_binary_operation(
    f: Callable[[Feature, Feature], Feature]
) -> Callable[[Any, Any], Feature]:
    """A decorator that enables binary operations to handle constant values.

    This decorator ensures that if one operand in a binary operation is a
    constant, it is converted into :class:`Feature` objects before the operation
    is performed. It wraps the provided binary operation function and seamlessly
    supports constants as inputs.

    Args:
        f (Callable[[Feature, Feature], Feature]): The binary operation function
            that operates on two :class:`Feature` objects.

    Returns:
        Callable[[Any, Any], Feature]: A wrapped function that handles constants
        by converting them into :class:`Feature` objects when necessary, then performs
        the binary operation.
    """

    def wrapper(a: Any, b: Any) -> Feature:
        # one must be a feature and if its a feature then the reference should be concrete
        assert isinstance(a, Feature) or isinstance(b, Feature)
        assert not isinstance(b, Feature) or isinstance(b.ref, ConcreteReference)
        assert not isinstance(a, Feature) or isinstance(a.ref, ConcreteReference)
        # add constant if required and call function
        return f(
            a if isinstance(a, Feature) else add_constant(a, b.dtype, b.ref._builder),
            b if isinstance(b, Feature) else add_constant(b, a.dtype, a.ref._builder),
        )

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
        FeatureResolver(
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
        return pc.negate(pc.cast(x, ctx.output_dtype.arrow_type))


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
        FeatureResolver(
            lambda _, inputs, session: (
                ScalarType
                if (
                    isinstance(inputs["x"], (Float32Feature, Float64Feature))
                    or isinstance(inputs["y"], (Float32Feature, Float64Feature))
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
            pc.cast(x, ctx.output_dtype.arrow_type), pc.cast(y, ctx.output_dtype.arrow_type)
        )


class FloorDivConfig(BaseDataProcessorConfig):
    """Configuration for the floor division processor."""


class FloorDiv(BaseDataProcessor[FloorDivConfig]):
    """Data Processor implementing floor division.

    This processor takes two numeric inputs, :code:`x` and :code:`y`, and
    computes their integer quotient (:code:`x // y`).
    """

    @process_mode(batched=True, backend="arrow")
    def process(
        self, ctx: RunContext, x: ScalarType, y: ScalarType
    ) -> Annotated[
        Int | UInt,
        FeatureResolver(
            lambda _, inputs, session: (
                ScalarType
                if (
                    isinstance(
                        inputs["x"],
                        (
                            Int8Feature,
                            Int16Feature,
                            Int32Feature,
                            Int64Feature,
                            UInt8Feature,
                            UInt16Feature,
                            UInt32Feature,
                            UInt64Feature,
                        ),
                    )
                    and isinstance(
                        inputs["y"],
                        (
                            Int8Feature,
                            Int16Feature,
                            Int32Feature,
                            Int64Feature,
                            UInt8Feature,
                            UInt16Feature,
                            UInt32Feature,
                            UInt64Feature,
                        ),
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
        return pc.cast(pc.floor(pc.divide(x, y)), ctx.output_dtype.arrow_type)


class MinConfig(BaseDataAggregatorConfig):
    """Configuration for the Min aggregator."""

    default: int | float = 0
    """The default initial value for the minimum."""


class Min(BaseDataAggregator[MinConfig]):
    """Data Aggregator implementing minimum value computation."""

    def seed(self, ctx: RunContext) -> tuple[ScalarType, bool]:
        """Initialize the aggregator's state.

        Args:
            ctx (RunContext): The execution context for the aggregator.

        Returns:
            tuple[ScalarType, bool]: A tuple containing the initial state value
            (default minimum) and a boolean indicating whether the state is initialized.
        """
        return self.config.default, True

    @process_mode(batched=True, backend="arrow")
    async def extract(self, ctx: RunContext, val: ScalarType) -> ScalarType:
        """Extract the minimum value from a batch of inputs.

        Args:
            ctx (RunContext): The execution context for the aggregator.
            val (ScalarType): A batch of numeric values.

        Returns:
            ScalarType: The minimum value from the input batch.
        """
        return pc.min(val).as_py()

    @process_mode(batched=False, backend="python")
    async def update(
        self, ctx: RunContext, val: ScalarType, state: bool, extracted: ScalarType
    ) -> tuple[ScalarType, bool]:
        """Update the aggregator's state with a new value.

        Args:
            ctx (RunContext): The execution context for the aggregator.
            val (ScalarType): The current state value.
            state (bool): A boolean indicating whether the state is initialized.
            extracted (ScalarType): The extracted value from the input.

        Returns:
            tuple[ScalarType, bool]: A tuple containing the updated state value
            (minimum) and a boolean indicating whether the state is initialized.
        """
        return extracted if state else min(extracted, val), False


class MaxConfig(BaseDataAggregatorConfig):
    """Configuration for the Max aggregator."""

    default: int | float = 0
    """The default initial value for the maximum."""


class Max(BaseDataAggregator[MaxConfig]):
    """Data Aggregator implementing maximum value computation."""

    def seed(self, ctx: RunContext) -> tuple[ScalarType, bool]:
        """Initialize the aggregator's state.

        Args:
            ctx (RunContext): The execution context for the aggregator.

        Returns:
            tuple[ScalarType, bool]: A tuple containing the initial state value
            (default maximum) and a boolean indicating whether the state is initialized.
        """
        return self.config.default, True

    @process_mode(batched=True, backend="arrow")
    async def extract(self, ctx: RunContext, val: ScalarType) -> ScalarType:
        """Extract the maximum value from a batch of inputs.

        Args:
            ctx (RunContext): The execution context for the aggregator.
            val (ScalarType): A batch of numeric values.

        Returns:
            ScalarType: The maximum value from the input batch.
        """
        return pc.max(val).as_py()

    @process_mode(batched=False, backend="python")
    async def update(
        self, ctx: RunContext, val: ScalarType, state: bool, extracted: ScalarType
    ) -> tuple[ScalarType, bool]:
        """Update the aggregator's state with a new value.

        Args:
            ctx (RunContext): The execution context for the aggregator.
            val (ScalarType): The current state value.
            state (bool): A boolean indicating whether the state is initialized.
            extracted (ScalarType): The extracted value from the input.

        Returns:
            tuple[ScalarType, bool]: A tuple containing the updated state value
            (maximum) and a boolean indicating whether the state is initialized.
        """
        return extracted if state else max(extracted, val), False


class SumConfig(BaseDataAggregatorConfig):
    """Configuration for the Sum aggregator."""

    start: int | float = 0
    """The default initial value for the sum."""


class Sum(BaseDataAggregator[SumConfig]):
    """Data Aggregator implementing sum value computation."""

    def seed(self, ctx: RunContext) -> tuple[ScalarType, None]:
        """Initialize the aggregator's state.

        Args:
            ctx (RunContext): The execution context for the aggregator.

        Returns:
            tuple[ScalarType, None]: A tuple containing the initial state value
            (default sum) and the unused aggregation state.
        """
        return self.config.start, None

    @process_mode(batched=True, backend="arrow")
    async def extract(self, ctx: RunContext, val: ScalarType) -> ScalarType:
        """Extract the sum value from a batch of inputs.

        Args:
            ctx (RunContext): The execution context for the aggregator.
            val (ScalarType): A batch of numeric values.

        Returns:
            ScalarType: The sum of the values in the input batch.
        """
        return pc.sum(val).as_py()

    @process_mode(batched=False, backend="python")
    async def update(
        self, ctx: RunContext, val: ScalarType, state: None, extracted: ScalarType
    ) -> tuple[ScalarType, None]:
        """Update the aggregator's state with a new value.

        Args:
            ctx (RunContext): The execution context for the aggregator.
            val (ScalarType): The current state value.
            state (None): Aggregation state, unused for summation.
            extracted (ScalarType): The extracted value from the input.

        Returns:
            tuple[ScalarType, None]: A tuple containing the updated value
            (sum) and the state (None).
        """
        return extracted + val, None


class MeanConfig(BaseDataAggregatorConfig):
    """Configuration for the Mean aggregator."""

    start: int | float = 0
    """The default initial value for the mean calculation."""

    start_count: int = 0
    """The initial count of values processed, used for mean calculation."""


class Mean(BaseDataAggregator[MeanConfig]):
    """Data Aggregator implementing the mean (average) calculation."""

    StateType: TypeAlias = tuple[ScalarType, int]

    def seed(self, ctx: RunContext) -> tuple[ScalarType, StateType]:
        """Initializes the state for the mean calculation.

        Args:
            ctx (RunContext): The execution context for the aggregator.

        Returns:
            tuple[ScalarType, StateType]: The initial state, which includes the
            starting sum and count of values.
        """
        return self.config.start / max(1, self.config.start_count), (
            self.config.start,
            self.config.start_count,
        )

    @process_mode(batched=True, backend="arrow")
    async def extract(self, ctx: RunContext, val: ScalarType) -> StateType:
        """Extracts the values to be aggregated.

        Args:
            ctx (RunContext): The execution context for the aggregator.
            val (ScalarType): The current value to be processed.

        Returns:
            StateType: The current sum and count for aggregation.
        """
        return pc.sum(val).as_py(), len(ctx.index)

    @process_mode(batched=False, backend="python")
    async def update(
        self, ctx: RunContext, val: ScalarType, state: StateType, extracted: ScalarType
    ) -> tuple[
        Annotated[
            Float,
            FeatureResolver(
                lambda _, inputs, session: Float32Feature
                if isinstance(inputs["val"], Float32Feature)
                else Float64Feature
            ),
        ],
        StateType,
    ]:
        """Updates the state by computing the mean after processing a new value.

        Args:
            ctx (RunContext): The execution context for the aggregator.
            val (ScalarType): The current value to be processed.
            state (StateType): The current state (sum and count).
            extracted (ScalarType): The extracted sum and count from the previous step.

        Returns:
            tuple[Float, StateType]: The updated mean and the new state (sum and count).
        """
        state = (state[0] + extracted[0], state[1] + extracted[1])
        return state[0] / max(1, state[1]), state


register_all(
    "__abs__",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
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
        Float32Feature,
        Float64Feature,
    ],
)(handle_constant_for_binary_operation(Add().call))

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
        Float32Feature,
        Float64Feature,
    ],
)(handle_constant_for_binary_operation(Subtract().call))

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
        Float32Feature,
        Float64Feature,
    ],
)(handle_constant_for_binary_operation(Multiply().call))

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
        Float32Feature,
        Float64Feature,
    ],
)(handle_constant_for_binary_operation(TrueDiv().call))

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
        Float32Feature,
        Float64Feature,
    ],
)(handle_constant_for_binary_operation(FloorDiv().call))


@register_all(
    "min",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
        Float32Feature,
        Float64Feature,
    ],
)
def _min(feature: Feature, default: int | float = 0) -> Feature:
    """Computes the minimum value of a feature.

    Args:
        feature (Feature): The feature whose minimum value is to be computed.
        default (int | float, optional): The default value to return if the
            feature is empty. Defaults to 0.

    Returns:
        Feature: A new feature representing the minimum value.
    """
    return Min(default=default).call(feature)


@register_all(
    "max",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
        Float32Feature,
        Float64Feature,
    ],
)
def _max(feature: Feature, default: int | float = 0) -> Feature:
    """Computes the maximum value of a feature.

    Args:
        feature (Feature): The feature whose maximum value is to be computed.
        default (int | float, optional): The default value to return if the
            feature is empty. Defaults to 0.

    Returns:
        Feature: A new feature representing the maximum value.
    """
    return Max(default=default).call(feature)


@register_all(
    "sum",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
        Float32Feature,
        Float64Feature,
    ],
)
def _sum(feature: Feature, start: int | float = 0) -> Feature:
    """Computes the sum of a feature.

    Args:
        feature (Feature): The feature whose values are to be summed.
        start (int | float, optional): The initial value to start the
            summation. Defaults to 0.

    Returns:
        Feature: A new feature representing the sum of the values.
    """
    return Sum(start=start).call(feature)


@register_all(
    "mean",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
        Float32Feature,
        Float64Feature,
    ],
)
def mean(feature: Feature, start: int | float = 0, start_count: int = 0) -> Feature:
    """Computes the mean (average) value of a feature.

    Args:
        feature (Feature): The feature whose mean value is to be computed.
        start (int | float, optional): The initial sum value to include in the
            computation. Defaults to 0.
        start_count (int, optional): The initial count of values to include in
            the computation. Defaults to 0.

    Returns:
        Feature: A new feature representing the mean value.
    """
    return Mean(start=start, start_count=start_count).call(feature)
