"""This module defines a collection of data processors for basic Boolean operations.

These processors allow for common Boolean logic operations such as inversion, logical
conjunction (AND), disjunction (OR), and exclusive disjunction (XOR). The processors
are designed to operate efficiently in a batched manner, leveraging the Apache Arrow
backend for optimized performance.

These processors are registered as methods on the `BoolFeature` class, enabling direct
application to Boolean features.
"""

from typing import TypeVar

import pyarrow.compute as pc

from ..features.features import BoolFeature, Feature
from ..nodes.base import RunContext, process_mode
from ..nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from ..typing import Annotated, Bool, MatchFeatures


class InvertConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`Invert` processor."""


class Invert(BaseDataProcessor[InvertConfig]):
    """Data processor for inverting Boolean values (logical NOT)."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, x: Bool) -> BoolFeature:
        """Inverts the input Boolean value.

        Args:
            ctx (RunContext): The execution context.
            x (Bool): The Boolean value to invert.

        Returns:
            BoolFeature: The inverted Boolean value.
        """
        return pc.invert(x)


class AndConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`And` processor."""


class And(BaseDataProcessor[AndConfig]):
    """Data processor for computing the logical AND of two Boolean values."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, a: Bool, b: Bool) -> BoolFeature:
        """Computes the logical AND of two Boolean values.

        Args:
            ctx (RunContext): The execution context.
            a (Bool): The first Boolean value.
            b (Bool): The second Boolean value.

        Returns:
            BoolFeature: The result of the logical AND operation.
        """
        return pc.and_(a, b)


class OrConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`Or` processor."""


class Or(BaseDataProcessor[OrConfig]):
    """Data processor for computing the logical OR of two Boolean values."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, a: Bool, b: Bool) -> BoolFeature:
        """Computes the logical OR of two Boolean values.

        Args:
            ctx (RunContext): The execution context.
            a (Bool): The first Boolean value.
            b (Bool): The second Boolean value.

        Returns:
            BoolFeature: The result of the logical OR operation.
        """
        return pc.or_(a, b)


class XorConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`Xor` processor."""


class Xor(BaseDataProcessor[XorConfig]):
    """Data processor for computing the logical XOR of two Boolean values."""

    @process_mode(batched=True, backend="arrow")
    def process(self, ctx: RunContext, a: Bool, b: Bool) -> BoolFeature:
        """Computes the logical XOR of two Boolean values.

        Args:
            ctx (RunContext): The execution context.
            a (Bool): The first Boolean value.
            b (Bool): The second Boolean value.

        Returns:
            BoolFeature: The result of the logical XOR operation.
        """
        return pc.xor(a, b)


class WhereConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`Where` processor."""


class Where(BaseDataProcessor[WhereConfig]):
    """Data processor that selects between two values based on a Boolean condition.

    This processor returns the first value if the condition is true,
    otherwise it returns the second value. It acts similarly to the
    ternary expression `a if cond else b`.
    """

    T = Annotated[TypeVar("T", bound=Feature), MatchFeatures()]

    @process_mode(batched=False, backend="python")
    def process(self, ctx: RunContext, cond: Bool, a: T, b: T) -> T:
        """Selects one of two values based on a Boolean condition.

        Args:
            ctx (RunContext): The execution context.
            cond (Bool): The Boolean condition to evaluate.
            a (T): The value to return if `cond` is True.
            b (T): The value to return if `cond` is False.

        Returns:
            T: The selected value based on the condition.
        """
        return a if cond else b


# Register all methods
BoolFeature.register_method("where")(Where().call)
BoolFeature.register_method("__invert__")(Invert().call)
BoolFeature.register_method("__and__")(BoolFeature.register_method("__rand__")(And().call))
BoolFeature.register_method("__or__")(BoolFeature.register_method("__ror__")(Or().call))
BoolFeature.register_method("__xor__")(BoolFeature.register_method("__rxor__")(Xor().call))
