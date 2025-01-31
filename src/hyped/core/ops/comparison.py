"""This module defines a collection of data processors for basic comparison operations.

These processors allow for fundamental comparison operations such as equality checks.
The processors are designed to operate efficiently in a batched manner, ensuring optimized
performance when handling feature comparisons.

These processors are registered as methods on the `Feature` class, enabling direct
application to feature objects.
"""

from typing import Annotated, TypeVar

from ..features.features import (
    BoolFeature,
    ClassLabelFeature,
    Feature,
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
from ..features.validators import MatchFeatures
from ..nodes.base import RunContext
from ..nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from ..typing import Bool


class EqualsConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`Equals` processor."""


T = TypeVar("T", bound=Feature)
Feat = Annotated[T, MatchFeatures()]


class Equals(BaseDataProcessor[EqualsConfig]):
    """Data processor for checking equality between two feature values."""

    def process(self, ctx: RunContext, obj1: Feat, obj2: Feat) -> Bool:
        """Computes whether two feature values are equal.

        Args:
            ctx (RunContext): The execution context.
            obj1 (Feat): The first feature value.
            obj2 (Feat): The second feature value.

        Returns:
            Bool: `True` if the feature values are equal, otherwise `False`.
        """
        return obj1 == obj2


class NotEqualsConfig(BaseDataProcessorConfig):
    """Configuration for the :class:`NotEquals` processor."""


class NotEquals(BaseDataProcessor[NotEqualsConfig]):
    """Data processor for checking inverse equality between two feature values."""

    def process(self, ctx: RunContext, obj1: Feat, obj2: Feat) -> Bool:
        """Computes whether two feature values are not equal.

        Args:
            ctx (RunContext): The execution context.
            obj1 (Feat): The first feature value.
            obj2 (Feat): The second feature value.

        Returns:
            Bool: `False` if the feature values are equal, otherwise `True`.
        """
        return obj1 != obj2


BoolFeature.register_method("__eq__")(Equals().call)
StringFeature.register_method("__eq__")(Equals().call)
Int8Feature.register_method("__eq__")(Equals().call)
Int16Feature.register_method("__eq__")(Equals().call)
Int32Feature.register_method("__eq__")(Equals().call)
Int64Feature.register_method("__eq__")(Equals().call)
UInt8Feature.register_method("__eq__")(Equals().call)
UInt16Feature.register_method("__eq__")(Equals().call)
UInt32Feature.register_method("__eq__")(Equals().call)
UInt64Feature.register_method("__eq__")(Equals().call)
Float32Feature.register_method("__eq__")(Equals().call)
Float64Feature.register_method("__eq__")(Equals().call)
MappingFeature.register_method("__eq__")(Equals().call)
SequenceFeature.register_method("__eq__")(Equals().call)
ClassLabelFeature.register_method("__eq__")(Equals().call)
